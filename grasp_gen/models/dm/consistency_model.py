import torch, pdb
from models.base import DIFFUSER
import torch.nn as nn
from omegaconf import DictConfig
import copy
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
from models.dm.schedule import make_schedule_ddpm
import numpy as np
import os
import trimesh
from utils.plotly_utils import plot_mesh
from plotly import graph_objects as go
import copy, math, peft
from einops import rearrange
from utils.handmodel import get_handmodel, pen_loss, spen_loss, dis_loss

def predicted_origin(model_output, timesteps, sample, alphas, sigmas):
    
    sigmas = extract_into_tensor(sigmas, timesteps, sample.shape)
    alphas = extract_into_tensor(alphas, timesteps, sample.shape)
    pred_x_0 = (sample - sigmas * model_output) / alphas

    return pred_x_0

def get_sigmas_karras(steps, num_inference_steps, device=None):

    lcm_timesteps = np.asarray(list(range(0, int(steps))))
    skipping_step = steps // num_inference_steps

    lcm_timesteps = lcm_timesteps[::-1].copy()
    inference_indices = np.linspace(0, len(lcm_timesteps), num=num_inference_steps, endpoint=False)
    inference_indices = np.floor(inference_indices).astype(np.int64)
    timesteps = lcm_timesteps[inference_indices]
    timesteps = torch.from_numpy(timesteps).to(device=device, dtype=torch.long)

    return timesteps

def make_master_params(param_groups_and_shapes):
    """
    Copy model parameters into a (differently-shaped) list of full-precision
    parameters.
    """
    master_params = []
    for param_group, shape in param_groups_and_shapes:
        master_param = nn.Parameter(
            _flatten_dense_tensors(
                [param.detach().float() for (_, param) in param_group]
            ).view(shape)
        )
        master_param.requires_grad = True
        master_params.append(master_param)
    return master_params

def unflatten_master_params(param_group, master_param):
    return _unflatten_dense_tensors(master_param, [param for (_, param) in param_group])

def master_params_to_model_params(param_groups_and_shapes, master_params):
    """
    Copy the master parameter data back into the model parameters.
    """
    # Without copying to a list, if a generator is passed, this will
    # silently not copy any parameters.
    for master_param, (param_group, _) in zip(master_params, param_groups_and_shapes):
        for (_, param), unflat_master_param in zip(
            param_group, unflatten_master_params(param_group, master_param.view(-1))
        ):
            param.detach().copy_(unflat_master_param)

def get_param_groups_and_shapes(named_model_params):
    named_model_params = list(named_model_params)
    scalar_vector_named_params = (
        [(n, p) for (n, p) in named_model_params if p.ndim <= 1],
        (-1),
    )
    matrix_named_params = (
        [(n, p) for (n, p) in named_model_params if p.ndim > 1],
        (1, -1),
    )
    return [scalar_vector_named_params, matrix_named_params]

def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(
            f"input has {x.ndim} dims but target_dims is {target_dims}, which is less"
        )
    return x[(...,) + (None,) * dims_to_append]

def get_weightings(weight_schedule, snrs, sigma_data):
    if weight_schedule == "snr":
        weightings = snrs
    elif weight_schedule == "snr+1":
        weightings = snrs + 1
    elif weight_schedule == "karras":
        weightings = snrs + 1.0 / sigma_data**2
    elif weight_schedule == "truncated-snr":
        weightings = torch.clamp(snrs, min=1.0)
    elif weight_schedule == "uniform":
        weightings = torch.ones_like(snrs)
    else:
        raise NotImplementedError()
    return weightings

def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))

def update_ema(target_params, source_params, rate=0.99):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.

    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)

def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

class DDPMSolver(nn.Module):
    def __init__(self, cfg: DictConfig, timesteps=100):
        # DDIM sampling parameters
        super(DDPMSolver, self).__init__()

        for k, v in make_schedule_ddpm(timesteps, **cfg.schedule_cfg).items():
            self.register_buffer(k, v)

        self.alpha_schedule = torch.sqrt(self.alphas_cumprod)
        self.sigma_schedule = torch.sqrt(1 - self.alphas_cumprod)

        self.ddpm_timesteps = (np.arange(1, timesteps + 1)).round().astype(np.int64) - 1
        self.ddpm_alpha_cumprods = self.alphas_cumprod
        self.ddpm_alpha_cumprods_prev = self.alphas_cumprod_prev
        self.final_alpha_cumprod = torch.tensor(1.0)
        # self.ddpm_alpha_cumprods_prev = np.asarray(
        #     [self.alphas_cumprod[0]] + self.alphas_cumprod[self.ddpm_timesteps[:-1]].tolist()
        # )
        # convert to torch tensors
        self.ddpm_timesteps = torch.from_numpy(self.ddpm_timesteps).long()
        # self.ddpm_alpha_cumprods = torch.from_numpy(self.ddpm_alpha_cumprods)
        # self.ddpm_alpha_cumprods_prev = torch.from_numpy(self.ddpm_alpha_cumprods_prev)

    def to_device(self, device):
        self.alpha_schedule = self.alpha_schedule.to(device)
        self.sigma_schedule = self.sigma_schedule.to(device)

        self.ddpm_timesteps = self.ddpm_timesteps.to(device)
        self.ddpm_alpha_cumprods = self.ddpm_alpha_cumprods.to(device)
        self.ddpm_alpha_cumprods_prev = self.ddpm_alpha_cumprods_prev.to(device)
        return self

    def ddpm_step(self, pred_x0, pred_noise, timestep_index):
        alpha_cumprod_prev = extract_into_tensor(self.ddpm_alpha_cumprods_prev, timestep_index, pred_x0.shape)
        dir_xt = (1.0 - alpha_cumprod_prev).sqrt() * pred_noise
        x_prev = alpha_cumprod_prev.sqrt() * pred_x0 + dir_xt
        return x_prev

@DIFFUSER.register()
class Consistency_model(nn.Module):
    def __init__(self, cfg: DictConfig, has_obser: bool, model: None, teacher_model: None, target_model:None, **kwargs):
        super(Consistency_model, self).__init__()
        self.sigma_data = cfg.sigma_data
        self.sigma_max = cfg.sigma_max
        self.sigma_min = cfg.sigma_min
        self.weight_schedule = cfg.weight_schedule
        self.distillation = cfg.distillation
        self.loss_norm = cfg.loss_norm
        self.rho = cfg.rho
        self._NORMALIZE_LOWER = -1.
        self._NORMALIZE_UPPER = 1.
        self._global_trans_lower = torch.tensor([-0.13128923, -0.10665303, -0.45753425], device='cuda')
        self._global_trans_upper = torch.tensor([0.12772022, 0.22954416, -0.21764427], device='cuda')
        # self.num_timesteps = 40
        self._joint_angle_lower = torch.tensor([-0.5235988, -0.7853982, -0.43633232, 0., 0., 0., -0.43633232, 0., 0., 0.,
                                       -0.43633232, 0., 0., 0., 0., -0.43633232, 0., 0., 0., -1.047, 0., -0.2618,
                                       -0.5237, 0.], device='cuda')
        self._joint_angle_upper = torch.tensor([0.17453292, 0.61086524, 0.43633232, 1.5707964, 1.5707964, 1.5707964, 0.43633232,
                                       1.5707964, 1.5707964, 1.5707964, 0.43633232, 1.5707964, 1.5707964, 1.5707964,
                                       0.6981317, 0.43633232, 1.5707964, 1.5707964, 1.5707964, 1.047, 1.309, 0.2618,
                                       0.5237, 1.], device='cuda')

        self.num_inference_steps = cfg.num_inference_steps
        self.rand_t_type = cfg.rand_t_type
        self.timesteps = cfg.timesteps
        self.device = 'cuda'
        self.solver = DDPMSolver(
            cfg,
            timesteps=cfg.timesteps,
        )
        self.eta = 1.0
        self.solver = self.solver.to(self.device)
        self.solver = self.solver.to_device(self.device)
        self.model = model
        self.teacher_model = teacher_model
        self.target_model = target_model
        self.master_params = list(self.model.parameters())
        self.target_ema = cfg.target_ema
        self.param_groups_and_shapes = get_param_groups_and_shapes(
            self.model.named_parameters()
        )
        self.master_params = make_master_params(self.param_groups_and_shapes)
        # self.hand_model = get_handmodel(batch_size = B, device = self.device)
        # pdb.set_trace()
        if self.target_model:
            self.target_model.requires_grad_(False)
            self.target_model.train()
            self.target_model_param_groups_and_shapes = get_param_groups_and_shapes(
                self.target_model.named_parameters()
            )
            self.target_model_master_params = make_master_params(
                self.target_model_param_groups_and_shapes
            )
        if teacher_model:
            self.teacher_model.requires_grad_(False)
            self.teacher_model.eval()
        self.optimizer = cfg.optimizer
        self.ema_rate = (
            [cfg.ema_rate]
            if isinstance(cfg.ema_rate, float)
            else [float(x) for x in cfg.ema_rate.split(",")]
        )
        self.ema_params = [
            copy.deepcopy(self.master_params)
            for _ in range(len(self.ema_rate))
        ]
    
    def get_snr(self, sigmas):
        return sigmas**-2

    def get_sigmas(self, sigmas):
        return sigmas

    def get_scalings(self, sigma):
        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2) ** 0.5
        c_in = 1 / (sigma**2 + self.sigma_data**2) ** 0.5
        return c_skip, c_out, c_in

    def get_scalings_for_boundary_condition(self, sigma):
        c_skip = self.sigma_data**2 / (
            (sigma / 0.1) ** 2 + self.sigma_data**2
        )
        c_out = (
            (sigma / 0.1) / ((sigma / 0.1)**2 + self.sigma_data**2) ** 0.5
        )
        c_in = 1 / (sigma**2 + self.sigma_data**2) ** 0.5
        return c_skip, c_out, c_in
    
    def denoise(self, model, x_t, sigmas, condtion):
        import torch.distributed as dist

        if not self.distillation:
            c_skip, c_out, c_in = [
                append_dims(x, x_t.ndim) for x in self.get_scalings(sigmas)
            ]
        else:
            c_skip, c_out, c_in = [
                append_dims(x, x_t.ndim)
                for x in self.get_scalings_for_boundary_condition(sigmas)
            ]
        # rescaled_t = 1000 * 0.25 * torch.log(sigmas + 1e-44)
        
        model_output = model(x_t, sigmas, condtion)
        denoised = c_out * model_output + c_skip * x_t
        return model_output, denoised
    
    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """ Forward difussion process, $q(x_t \mid x_0)$, this process is determinative 
        and has no learnable parameters.

        $x_t = \sqrt{\bar{\alpha}_t} * x0 + \sqrt{1 - \bar{\alpha}_t} * \epsilon$

        """
        B, *x_shape = x0.shape
        
        x_t = self.solver.sqrt_alphas_cumprod[t].reshape(B, *((1, ) * len(x_shape))) * x0 + \
            self.solver.sqrt_one_minus_alphas_cumprod[t].reshape(B, *((1, ) * len(x_shape))) * noise

        return x_t

    def consistency_losses(self, data):
        x_start = data['x']
        B, *x_shape = x_start.shape
        # if model_kwargs is None:
        #     model_kwargs = {}
        noise = torch.randn_like(x_start)

        dims = x_start.ndim

        # indices = torch.randint(
        #     0, num_scales - 1, (x_start.shape[0],), device=x_start.device
        # )
        
        condtion = self.model.eps_model.condition(data)

        index = torch.randint(0, self.timesteps, (x_start.shape[0],), device=self.device).long()
        ts = self.solver.ddpm_timesteps[index].to(self.device)
        
        pre_timesteps = ts - 1
        pre_timesteps = torch.where(pre_timesteps < 0, torch.zeros_like(pre_timesteps), pre_timesteps)

        c_skip_start, c_out_start, c_in = self.get_scalings_for_boundary_condition(ts)
        c_skip_start, c_out_start = [append_dims(x, x_start.ndim) for x in [c_skip_start, c_out_start]]
        c_skip, c_out, _ = self.get_scalings_for_boundary_condition(pre_timesteps)
        c_skip, c_out = [append_dims(x, x_start.ndim) for x in [c_skip, c_out]]
        
        x_t = self.q_sample(x0=data['x'], t=ts, noise=noise)
        
        noise_pred = self.model.eps_model(x_t, ts, condtion)
        
        # pred_x_0 = predicted_origin(
        #     noise_pred,
        #     ts,
        #     x_t,
        #     self.solver.alpha_schedule,
        #     self.solver.sigma_schedule,
        # )
        pred_x_0 = self.solver.sqrt_recip_alphas_cumprod[ts].reshape(B, *((1, ) * len(x_shape))) * x_t - \
            self.solver.sqrt_recipm1_alphas_cumprod[ts].reshape(B, *((1, ) * len(x_shape))) * noise_pred
        '''
        ## vis x_0
        x_0 = pred_x_0
        x_0[:, :3] = self.trans_denormalize(x_0[:, :3])
        x_0[:, 9:] = self.angle_denormalize(x_0[:, 9:])
        self.visualize_(x_0[:4, :], data['scene_id'][:4])
        pdb.set_trace()
        '''
        model_pred = c_skip_start * x_t + c_out_start * pred_x_0

        with torch.no_grad():
            with torch.autocast("cuda"):
                teacher_noise_pred = self.teacher_model.eps_model(x_t, ts, condtion)
                teacher_pred_x0 = self.solver.sqrt_recip_alphas_cumprod[ts].reshape(B, *((1, ) * len(x_shape))) * x_t - \
                    self.solver.sqrt_recipm1_alphas_cumprod[ts].reshape(B, *((1, ) * len(x_shape))) * teacher_noise_pred
                '''
                ## vis teacher_pred_x0
                teacher_pred_x0[:, :3] = self.trans_denormalize(teacher_pred_x0[:, :3])
                teacher_pred_x0[:, 9:] = self.angle_denormalize(teacher_pred_x0[:, 9:])
                self.visualize_(teacher_pred_x0[:1, :], data['scene_id'][:1])
                teacher_pred_x0[:, :3] = self.trans_normalize(teacher_pred_x0[:, :3])
                teacher_pred_x0[:, 9:] = self.angle_normalize(teacher_pred_x0[:, 9:])
                '''
                # pdb.set_trace()
                x_prev = self.solver.ddpm_step(teacher_pred_x0, teacher_noise_pred, index)
                '''
                ## vis x_prev
                x_0 = x_prev
                x_0[:, :3] = self.trans_denormalize(x_0[:, :3])
                x_0[:, 9:] = self.angle_denormalize(x_0[:, 9:])
                self.visualize_(x_0[:1, :], data['scene_id'][:1])
                pdb.set_trace()
                '''
        with torch.no_grad():
            with torch.autocast("cuda"):
                target_noise_pred = self.target_model.eps_model(x_prev, pre_timesteps, condtion)
                target_pred_x0 = self.solver.sqrt_recip_alphas_cumprod[pre_timesteps].reshape(B, *((1, ) * len(x_shape))) * x_prev - \
                    self.solver.sqrt_recipm1_alphas_cumprod[pre_timesteps].reshape(B, *((1, ) * len(x_shape))) * target_noise_pred
                '''
                ## vis x_prev
                x_0 = target_pred_x0
                x_0[:, :3] = self.trans_denormalize(x_0[:, :3])
                x_0[:, 9:] = self.angle_denormalize(x_0[:, 9:])
                self.visualize_(x_0[:1, :], data['scene_id'][:1])
                pdb.set_trace()
                '''
                target = c_skip * x_prev + c_out * target_pred_x0
        # pdb.set_trace()
        if self.optimizer:
            hand_model = get_handmodel(batch_size = B, device = self.device)
            pred_x_0[:, :3] = self.trans_denormalize(pred_x_0[:, :3])
            if pred_x_0[:, 3:].size(1) == 24:
                pred_x_0[:, 3:] = self.angle_denormalize(pred_x_0[:, 3:])
            else:
                pred_x_0[:, 9:] = self.angle_denormalize(pred_x_0[:, 9:])
            
            pred_x_0 = torch.cat([pred_x_0[:, :3], pred_x_0[:, 3:]], dim=-1)
            obj_pcd = data['pos'].to(self.device)
            hand_model.update_kinematics(q = pred_x_0)
            'TODO:use penloss'
            hand_pcd = hand_model.get_surface_points(q= pred_x_0).to(dtype=torch.float32)
            normal = torch.tensor(np.array(data['normal']), device=self.device)
            'TODO:use transformer'
            obj_pcd = rearrange(obj_pcd, '(b n) c -> b n c', b=B, n=normal.shape[1])
            obj_pcd_nor = torch.cat((obj_pcd, normal), dim=-1).to(dtype=torch.float32)
            pen_loss_value = pen_loss(obj_pcd_nor, hand_pcd)
            # print(f'pen_loss:{pen_loss_value}')
            "TODO:use disloss"
            dis_keypoint = hand_model.get_dis_keypoints(q= pred_x_0)
            dis_loss_value = dis_loss(dis_keypoint, obj_pcd)
            # print(f'dis_loss:{dis_loss_value}')
            'TODO:use spenloss'
            hand_keypoint = hand_model.get_keypoints(q= pred_x_0)
            spen_loss_value = spen_loss(hand_keypoint)
            # print(f'spen_loss:{spen_loss_value}')
            guidance_loss = pen_loss_value + dis_loss_value + spen_loss_value

        if self.loss_norm == "l1":
            diffs = torch.abs(model_pred - target)
            loss = mean_flat(diffs)
        elif self.loss_norm == "l2":
            diffs = (model_pred - target) ** 2
            loss = mean_flat(diffs)
        else:
            raise ValueError(f"Unknown loss norm {self.loss_norm}")

        terms = {}
        terms["guidance_loss"] = guidance_loss
        terms["loss"] = loss.mean() + guidance_loss

        print(f'loss_dfm:{terms["loss"]}')

        return terms
    
    def angle_normalize(self, joint_angle: torch.Tensor):
        joint_angle_norm = torch.div((joint_angle - self._joint_angle_lower), (self._joint_angle_upper - self._joint_angle_lower))
        joint_angle_norm = joint_angle_norm * (self._NORMALIZE_UPPER - self._NORMALIZE_LOWER) - (self._NORMALIZE_UPPER - self._NORMALIZE_LOWER) / 2
        return joint_angle_norm

    def angle_denormalize(self, joint_angle: torch.Tensor):
        if joint_angle.shape[-1] == 22:
            joint_angle_upper = self._joint_angle_upper[2:]
            joint_angle_lower = self._joint_angle_lower[2:]
        else:
            joint_angle_upper = self._joint_angle_upper
            joint_angle_lower = self._joint_angle_lower
        
        joint_angle_denorm = joint_angle + (self._NORMALIZE_UPPER - self._NORMALIZE_LOWER) / 2
        joint_angle_denorm /= (self._NORMALIZE_UPPER - self._NORMALIZE_LOWER)
        joint_angle_denorm = joint_angle_denorm * (joint_angle_upper - joint_angle_lower) + joint_angle_lower
        return joint_angle_denorm

    def trans_denormalize(self, global_trans: torch.Tensor):
        global_trans_denorm = global_trans + (self._NORMALIZE_UPPER - self._NORMALIZE_LOWER) / 2
        global_trans_denorm /= (self._NORMALIZE_UPPER - self._NORMALIZE_LOWER)
        global_trans_denorm = global_trans_denorm * (self._global_trans_upper - self._global_trans_lower) + self._global_trans_lower
        return global_trans_denorm
    
    def trans_normalize(self, global_trans: torch.Tensor):
        global_trans_norm = torch.div((global_trans - self._global_trans_lower), (self._global_trans_upper - self._global_trans_lower))
        global_trans_norm = global_trans_norm * (self._NORMALIZE_UPPER - self._NORMALIZE_LOWER) - (self._NORMALIZE_UPPER - self._NORMALIZE_LOWER) / 2
        return global_trans_norm

    def _update_ema(self):
        
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.master_params, rate=rate)
    
    def _update_target_ema(self):
        with torch.no_grad():
            # pdb.set_trace()
            update_ema(
                self.target_model_master_params,
                self.master_params,
                rate=self.target_ema,
            )
            # for params_tar, params_ in zip(self.target_model_master_params, self.master_params):
            #     update_ema(
            #         params_tar,
            #         self.master_params,
            #         rate=self.target_ema,
            #     )
            master_params_to_model_params(
                self.target_model_param_groups_and_shapes,
                self.target_model_master_params,
            )
    
    def sample_logprob(self, noise_pred, t, latents, prev_sample=None):
        
        B, *x_shape = latents.shape
        pred_x0 = self.solver.sqrt_recip_alphas_cumprod[t].reshape(B, *((1, ) * len(x_shape))) * latents - \
            self.solver.sqrt_recipm1_alphas_cumprod[t].reshape(B, *((1, ) * len(x_shape))) * noise_pred
        model_mean = self.solver.posterior_mean_coef1[t].reshape(B, *((1, ) * len(x_shape))) * pred_x0 + \
            self.solver.posterior_mean_coef2[t].reshape(B, *((1, ) * len(x_shape))) * latents
        posterior_variance = self.solver.posterior_variance[t].reshape(B, *((1, ) * len(x_shape)))
        # posterior_log_variance = self.posterior_log_variance_clipped[t].reshape(B, *((1, ) * len(x_shape))) # clipped variance
        std_dev_t = self.eta * posterior_variance.clamp(min=1e-5) ** (0.5)

        log_prob = (
        -((latents.detach() - model_mean) ** 2) / (2 * (std_dev_t**2))
        - torch.log(std_dev_t)
        - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
        )
        log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))

        return log_prob
    
    def index_for_timestep(self, timestep, schedule_timesteps=None):
        if schedule_timesteps is None:
            schedule_timesteps = self.timesteps

        indices = (schedule_timesteps == timestep).nonzero()

        pos = 1 if len(indices) > 1 else 0

        return indices[pos].item()

    def step_t_t_1(self, noise_pred, timestep, x_t, timesteps, data, step_index=None):
        B, *x_shape = x_t.shape
        step_index = self.index_for_timestep(timestep, timesteps)

        prev_step_index = step_index + 1
        if prev_step_index < len(timesteps):
            prev_timestep = timesteps[prev_step_index]
        else:
            prev_timestep = timestep

        timestep = torch.full((B, ), timestep, device=self.device, dtype=torch.long)
        prev_timestep = torch.full((B, ), prev_timestep, device=self.device, dtype=torch.long)

        alpha_prod_t = self.solver.alphas_cumprod[timestep]
        # alpha_prod_t_prev = self.solver.alphas_cumprod[prev_timestep]
        alpha_prod_t_prev = self.solver.alphas_cumprod_prev[timestep]             ##########
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        c_skip, c_out, c_in = self.get_scalings_for_boundary_condition(timestep)
        predicted_original_sample = self.solver.sqrt_recip_alphas_cumprod[timestep].reshape(B, *((1, ) * len(x_shape))) * x_t - \
            self.solver.sqrt_recipm1_alphas_cumprod[timestep].reshape(B, *((1, ) * len(x_shape))) * noise_pred
        '''
        ## vis x_prev
        pred_x0 = predicted_original_sample
        pred_x0[:, :3] = self.trans_denormalize(pred_x0[:, :3])
        pred_x0[:, 9:] = self.angle_denormalize(pred_x0[:, 9:])
        self.visualize_(pred_x0[:1, :], data['scene_id'][:1])
        pdb.set_trace()
        '''
        # model_mean = self.solver.posterior_mean_coef1[timestep].reshape(B, *((1, ) * len(x_shape))) * predicted_original_sample + \
        #     self.solver.posterior_mean_coef2[timestep].reshape(B, *((1, ) * len(x_shape))) * x_t
        # model_log_variance = self.solver.posterior_log_variance_clipped[timestep].reshape(B, *((1, ) * len(x_shape)))

        denoised = c_out.reshape(B, *((1, ) * len(x_shape))) * predicted_original_sample + c_skip.reshape(B, *((1, ) * len(x_shape))) * x_t
        # denoised = None
        if step_index != self.num_inference_steps - 1:
            noise = torch.randn_like(x_t)
            prev_sample = alpha_prod_t_prev.sqrt().reshape(B, *((1, ) * len(x_shape))) * denoised + beta_prod_t_prev.sqrt().reshape(B, *((1, ) * len(x_shape))) * noise
        else:
            prev_sample = denoised
        # prev_sample = model_mean + (0.5 * model_log_variance).exp() * noise
        
        return prev_sample

    def sample(self, data, k=None):
        
        B, *x_shape = data['x'].shape
        condtion = self.model.eps_model.condition(data)
        data['cond'] = condtion
        # if generator is None:
        #     generator = get_generator("dummy")
        
        timesteps = get_sigmas_karras(self.timesteps, self.num_inference_steps, device=self.device)
        # pdb.set_trace()
        x_t = data['x']

        # with torch.no_grad():
        all_x_t = [x_t]
        all_time = []
        all_log_probs = []
        for i, ts in enumerate(timesteps):
            batch_ts = torch.full((B, ), ts, device=self.device, dtype=torch.long)
            # if i == 31:
            #     pdb.set_trace()
            noise_pred = self.model.eps_model(x_t, batch_ts, condtion)
            
            x_t = self.step_t_t_1(noise_pred, ts, x_t, timesteps, data, step_index = i)
            all_x_t.append(x_t)
            all_time.append(ts)

        # pdb.set_trace()
        '''
        pred_x0 = all_x_t[-1]
        ## vis x_prev
        pred_x0[:, :3] = self.trans_denormalize(pred_x0[:, :3])
        pred_x0[:, 9:] = self.angle_denormalize(pred_x0[:, 9:])
        self.visualize_(pred_x0[:10, :], data['scene_id'][:10])
        '''

        return torch.stack(all_x_t, dim=0), None, torch.tensor(all_time, device = self.device)
    
    def visualize_(self, mdata_qpos, scene_object):

        for i in range(mdata_qpos.shape[0]):
            mesh_path = os.path.join('/inspurfs/group/mayuexin/datasets/Realdex/meshdata', f'{scene_object[i]}.obj')
            obj_mesh = trimesh.load(mesh_path)
            hand_model = get_handmodel(batch_size=1, device=self.device)
            hand_model.update_kinematics(q=mdata_qpos[i:i+1, :])
            vis_data = [plot_mesh(obj_mesh, color='lightpink')]
            
            vis_data += hand_model.get_plotly_data(opacity=1.0, color='#8799C6')
            if i <100:
                # 保存为 HTML 文件
                save_path = os.path.join('/inspurfs/group/mayuexin/zhuyufei/graps_gen/outputs/temp', f'{scene_object[i]}_sample-{i}.html')
                fig = go.Figure(data=vis_data)
                fig.update_layout(
                    scene=dict(
                            xaxis=dict(visible=False),
                            yaxis=dict(visible=False),
                            zaxis=dict(visible=False),
                            bgcolor="white"
                        )
                )
                fig.write_html(save_path)

        '''
        if self.sampler == 'onestep':
            sample_fn = self.sample_onestep
        

        sample_fn = {
            "heun": sample_heun,
            "dpm": sample_dpm,
            "ancestral": sample_euler_ancestral,
            "onestep": sample_onestep,
            "progdist": sample_progdist,
            "euler": sample_euler,
            "multistep": stochastic_iterative_sampler,
        }[self.sampler]

        if self.sampler in ["heun", "dpm"]:
            sampler_args = dict(
                s_churn=s_churn, s_tmin=s_tmin, s_tmax=s_tmax, s_noise=s_noise
            )
        elif self.sampler == "multistep":
            sampler_args = dict(
                ts=ts, t_min=sigma_min, t_max=sigma_max, rho=self.rho, steps=self.timesteps
            )
        else:
            sampler_args = {}

        def denoiser(x_t, sigma):
            
            _, denoised = self.denoise(self.model, x_t, sigma, condition)
            # if clip_denoised:
            denoised = denoised.clamp(-1, 1)
            return denoised
        
        all_x_0 = []
        x_0 = sample_fn(
            denoiser,
            x_T,
            sigmas,
            **sampler_args,
        )
        all_x_0.append(x_0)

        return torch.stack(all_x_0, dim=1), None, None
    
    def sample_onestep(
        self,
        distiller,
        x,
        sigmas,
    ):
        """Single-step generation from a distilled model."""
        all_x_t = []
        s_in = x.new_ones([x.shape[0]])
        x_0 = distiller(x, sigmas[0] * s_in)
        all_x_t.append(x_0)
        return torch.stack(all_x_t, dim=0)
        '''
