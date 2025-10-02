from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from utils.handmodel import get_handmodel, pen_loss, spen_loss, dis_loss
from models.base import DIFFUSER
from models.dm.schedule import make_schedule_ddpm ,get_beta_schedule
from models.optimizer.optimizer import Optimizer
from models.planner.planner import Planner
from plyfile import PlyData, PlyElement
import numpy as np
from einops import rearrange
import math, pdb, os
import trimesh
from utils.plotly_utils import plot_mesh
from plotly import graph_objects as go
import copy
import pickle

@DIFFUSER.register()
class DDPM(nn.Module):
    def __init__(self, eps_model: nn.Module, cfg: DictConfig, has_obser: bool, *args, **kwargs) -> None:
        super(DDPM, self).__init__()
        self._joint_angle_lower = torch.tensor([-0.5235988, -0.7853982, -0.43633232, 0., 0., 0., -0.43633232, 0., 0., 0.,
                                       -0.43633232, 0., 0., 0., 0., -0.43633232, 0., 0., 0., -1.047, 0., -0.2618,
                                       -0.5237, 0.], device='cuda')
        self._joint_angle_upper = torch.tensor([0.17453292, 0.61086524, 0.43633232, 1.5707964, 1.5707964, 1.5707964, 0.43633232,
                                       1.5707964, 1.5707964, 1.5707964, 0.43633232, 1.5707964, 1.5707964, 1.5707964,
                                       0.6981317, 0.43633232, 1.5707964, 1.5707964, 1.5707964, 1.047, 1.309, 0.2618,
                                       0.5237, 1.], device='cuda')
        
        self._global_trans_lower = torch.tensor([-0.13128923, -0.10665303, -0.45753425], device='cuda')
        self._global_trans_upper = torch.tensor([0.12772022, 0.22954416, -0.21764427], device='cuda')
        self._NORMALIZE_LOWER = -1.
        self._NORMALIZE_UPPER = 1.
        self.num_inference_steps = cfg.num_inference_steps
        self.eta = 1.0
        self.sigma_data = cfg.sigma_data
        self.cfg = cfg
        self.eps_model = eps_model
        self.timesteps = cfg.steps
        self.schedule_cfg = cfg.schedule_cfg
        self.rand_t_type = cfg.rand_t_type
        self.use_all_pra = cfg.use_all_pra_2
        self.has_observation = has_obser # used in some task giving observation
        # self.training_type = cfg.training_type
        self.timesteps_ = None
        for k, v in make_schedule_ddpm(self.timesteps, **self.schedule_cfg).items():
            self.register_buffer(k, v)

        if cfg.loss_type == 'l1':
            self.criterion = F.l1_loss
        elif cfg.loss_type == 'l2':
            self.criterion = F.mse_loss
        else:
            raise Exception('Unsupported loss type.')
        ## flow matching or short cut
        self.BOOTSTRAP_EVERY = 8
        self.DENOISE_TIMESTEPS = 128
        self.CLASS_DROPOUT_PROB = 1.0
        self.NUM_CLASSES = 1
        ##
        self.optimizer = None
        self.planner = None

    @property
    def device(self):
        return self.betas.device

    def get_sigmas_karras(self, steps, num_inference_steps, device=None):

        lcm_timesteps = np.asarray(list(range(0, int(steps))))
        skipping_step = steps // num_inference_steps

        lcm_timesteps = lcm_timesteps[::-1].copy()
        inference_indices = np.linspace(0, len(lcm_timesteps), num=num_inference_steps, endpoint=False)
        inference_indices = np.floor(inference_indices).astype(np.int64)
        timesteps = lcm_timesteps[inference_indices]
        timesteps = torch.from_numpy(timesteps).to(device=device, dtype=torch.long)

        return timesteps
    
    def get_scalings_for_boundary_condition(self, sigma):
        c_skip = self.sigma_data**2 / (
            (sigma / 0.1) ** 2 + self.sigma_data**2
        )
        c_out = (
            (sigma / 0.1) / ((sigma / 0.1)**2 + self.sigma_data**2) ** 0.5
        )
        c_in = 1 / (sigma**2 + self.sigma_data**2) ** 0.5
        return c_skip, c_out, c_in

    def apply_observation(self, x_t: torch.Tensor, data: Dict) -> torch.Tensor:
        """ Apply observation to x_t, if self.has_observation if False, this method will return the input

        Args:
            x_t: noisy x in step t
            data: original data provided by dataloader
        """
        ## has start observation, used in path planning and start-conditioned motion generation
        if self.has_observation and 'start' in data:
            start = data['start'] # <B, T, D>
            T = start.shape[1]
            x_t[:, 0:T, :] = start[:, 0:T, :].clone()
        
            if 'obser' in data:
                obser = data['obser']
                O = obser.shape[1]
                x_t[:, T:T+O, :] = obser.clone()
        
        return x_t
    
    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """ Forward difussion process, $q(x_t \mid x_0)$, this process is determinative 
        and has no learnable parameters.

        $x_t = \sqrt{\bar{\alpha}_t} * x0 + \sqrt{1 - \bar{\alpha}_t} * \epsilon$

        Args:
            x0: samples at step 0
            t: diffusion step
            noise: Gaussian noise
        
        Return:
            Diffused samples
        """
        B, *x_shape = x0.shape
        x_t = self.sqrt_alphas_cumprod[t].reshape(B, *((1, ) * len(x_shape))) * x0 + \
            self.sqrt_one_minus_alphas_cumprod[t].reshape(B, *((1, ) * len(x_shape))) * noise

        return x_t

    def forward(self, data: Dict) -> torch.Tensor:
        """ Reverse diffusion process, sampling with the given data containing condition

        Args:
            data: test data, data['x'] gives the target data, data['y'] gives the condition
        
        Return:
            Computed loss
        """
        
        B, *x_shape = data['x'].shape
        ## randomly sample timesteps

        if self.rand_t_type == 'all':
            ts = torch.randint(0, self.timesteps, (B, ), device=self.device).long()
        elif self.rand_t_type == 'half':
            ts = torch.randint(0, self.timesteps, ((B + 1) // 2, ), device=self.device)
            if B % 2 == 1:
                ts = torch.cat([ts, self.timesteps - ts[:-1] - 1], dim=0).long()
            else:
                ts = torch.cat([ts, self.timesteps - ts - 1], dim=0).long()
        else:
            raise Exception('Unsupported rand ts type.')
        
        ## generate Gaussian noise
        noise = torch.randn_like(data['x'], device=self.device)

        ## calculate x_t, forward diffusion process

        x_t = self.q_sample(x0=data['x'], t=ts, noise=noise)
        ## apply observation before forwarding to eps model
        ## model need to learn the relationship between the observation frames and the rest frames
        x_t = self.apply_observation(x_t, data)

        ## predict noise
        # pdb.set_trace()
        condtion = self.eps_model.condition(data)
        output = self.eps_model(x_t, ts, condtion)
        ## apply observation after forwarding to eps model
        ## this operation will detach gradient from the loss of the observation tokens
        ## because the target and output of the observation tokens all are constants
        output = self.apply_observation(output, data)
        pred_x0 = self.sqrt_recip_alphas_cumprod[ts].reshape(B, *((1, ) * len(x_shape))) * x_t - \
            self.sqrt_recipm1_alphas_cumprod[ts].reshape(B, *((1, ) * len(x_shape))) * output
        hand_model = get_handmodel(batch_size = B, device = self.device)
        pred_x0[:, :3] = self.trans_denormalize(pred_x0[:, :3])
        if self.use_all_pra:
            # print('2----------------------')
            pred_x0[:, 9:] = self.angle_denormalize(pred_x0[:, 9:])
            pred_x0 = torch.cat([pred_x0[:, :3], pred_x0[:, 3:]], dim=-1)
        else:
            id_6d_rot = torch.tensor([1., 0., 0., 0., 1., 0.], device = self.device).view(1, 6).repeat(B, 1)
            pred_x0[:, 3:] = self.angle_denormalize(pred_x0[:, 3:])
            pred_x0 = torch.cat([pred_x0[:, :3], id_6d_rot, pred_x0[:, 3:]], dim=-1)
        # pdb.set_trace()
        # self.visualize_(pred_x0[:1, :], data['scene_id'][:1])
        
        obj_pcd = data['pos'].to(self.device)

        hand_model.update_kinematics(q = pred_x0)

        'TODO:use penloss'
        hand_pcd = hand_model.get_surface_points(q= pred_x0).to(dtype=torch.float32)
        normal = torch.tensor(np.array(data['normal']), device=self.device)
        'TODO:use transformer'
        obj_pcd = rearrange(obj_pcd, '(b n) c -> b n c', b=B, n=normal.shape[1])
        obj_pcd_nor = torch.cat((obj_pcd, normal), dim=-1).to(dtype=torch.float32)
        pen_loss_value = pen_loss(obj_pcd_nor, hand_pcd)
        print(f'pen_loss:{pen_loss_value}')
        
        # self.save_point_cloud_with_normals(hand_pcd[:30, :, :], obj_pcd[:30, :, :], normal[:30, :, :])
        # self.visualize_(pred_x0[:30, :], data['scene_id'][:30])
        # pdb.set_trace()
        "TODO:use disloss"
        dis_keypoint = hand_model.get_dis_keypoints(q= pred_x0)
        dis_loss_value = dis_loss(dis_keypoint, obj_pcd)
        print(f'dis_loss:{dis_loss_value}')

        'TODO:use spenloss'
        hand_keypoint = hand_model.get_keypoints(q= pred_x0)
        spen_loss_value = spen_loss(hand_keypoint)
        print(f'spen_loss:{spen_loss_value}')

        ## calculate loss
        loss_dfm =self.criterion(output, noise)
        print(f'loss_dfm:{loss_dfm}')
        loss = loss_dfm  + spen_loss_value + dis_loss_value + pen_loss_value
        # self.save_point_cloud_with_normals(hand_pcd, obj_pcd, normal)
        return {'loss': loss}
    
    def visualize_(self, mdata_qpos, scene_object_, time_step, mark):
        res = {'method': 'diffuser@w/o-opt',
            'desc': 'w/o optimizer grasp pose generation',
            'sample_qpos': {}}

        for i in range(mdata_qpos.shape[0]):
            scene_dataset, scene_object = scene_object_[i].split('+')
            mesh_path = os.path.join('assets/object', scene_dataset, scene_object, f'{scene_object}.stl')
            # mesh_path = os.path.join('/inspurfs/group/mayuexin/datasets/Realdex/meshdata', f'{scene_object[i]}.obj')
            # mesh_path = os.path.join('/inspurfs/group/mayuexin/datasets/DexGRAB/contact_meshes', f'{scene_object[i]}.ply')
            obj_mesh = trimesh.load(mesh_path)
            hand_model = get_handmodel(batch_size=1, device=self.device)
            hand_model.update_kinematics(q=mdata_qpos[i:i+1, :])
            vis_data = [plot_mesh(obj_mesh, color='lightpink')]
            
            vis_data += hand_model.get_plotly_data(opacity=1.0, color='#8799C6')
            if i <100:
                # 保存为 HTML 文件
                if mark:
                    save_path = os.path.join('/inspurfs/group/mayuexin/zhuyufei/graps_gen/outputs/temp/x0', f'{i}_sample-{time_step}.html')
                else:
                    save_path = os.path.join('/inspurfs/group/mayuexin/zhuyufei/graps_gen/outputs/temp/gradient_pre', f'{i}_sample-{time_step}.html')
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
            res['sample_qpos'][scene_object] = np.array(mdata_qpos[i:i+1, :].cpu().detach())
            if mark:
                pickle.dump(res, open(os.path.join(f'/inspurfs/group/mayuexin/zhuyufei/graps_gen/outputs/paper/x0', f'{i}_res_diffuser_{time_step}.pkl'), 'wb'))
            else:
                pickle.dump(res, open(os.path.join(f'/inspurfs/group/mayuexin/zhuyufei/graps_gen/outputs/paper/gradient_pre', f'{i}_res_diffuser_{time_step}.pkl'), 'wb'))

    def model_predict(self, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> Tuple:
        """ Get and process model prediction

        $x_0 = \frac{1}{\sqrt{\bar{\alpha}_t}}(x_t - \sqrt{1 - \bar{\alpha}_t}\epsilon_t)$

        Args:
            x_t: denoised sample at timestep t
            t: denoising timestep
            cond: condition tensor
        
        Return:
            The predict target `(pred_noise, pred_x0)`, currently we predict the noise, which is as same as DDPM
        """
        B, *x_shape = x_t.shape

        pred_noise = self.eps_model(x_t, t, cond)
        pred_x0 = self.sqrt_recip_alphas_cumprod[t].reshape(B, *((1, ) * len(x_shape))) * x_t - \
            self.sqrt_recipm1_alphas_cumprod[t].reshape(B, *((1, ) * len(x_shape))) * pred_noise

        return pred_noise, pred_x0
    
    def p_mean_variance(self, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> Tuple:
        """ Calculate the mean and variance, we adopt the following first equation.

        $\tilde{\mu} = \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}x_t + \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1 - \bar{\alpha}_t}x_0$
        $\tilde{\mu} = \frac{1}{\sqrt{\alpha}_t}(x_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}}\epsilon_t)$

        Args:
            x_t: denoised sample at timestep t
            t: denoising timestep
            cond: condition tensor
        
        Return:
            (model_mean, posterior_variance, posterior_log_variance)
        """
        B, *x_shape = x_t.shape

        ## predict noise and x0 with model $p_\theta$
        # pdb.set_trace()
        pred_noise, pred_x0 = self.model_predict(x_t, t, cond)

        ## calculate mean and variance
        model_mean = self.posterior_mean_coef1[t].reshape(B, *((1, ) * len(x_shape))) * pred_x0 + \
            self.posterior_mean_coef2[t].reshape(B, *((1, ) * len(x_shape))) * x_t
        posterior_variance = self.posterior_variance[t].reshape(B, *((1, ) * len(x_shape)))
        posterior_log_variance = self.posterior_log_variance_clipped[t].reshape(B, *((1, ) * len(x_shape))) # clipped variance

        return model_mean, posterior_variance, posterior_log_variance, pred_x0, pred_noise

    # consistency model x_{t}-->x_{t-k}
    def step_t_t_1(self, noise_pred, timestep, x_t, data, timesteps, predicted_original_sample=None, step_index=None):
        B, *x_shape = x_t.shape

        prev_step_index = step_index + 1
        if prev_step_index < len(timesteps):
            prev_timestep = timesteps[prev_step_index]
            prev_timestep = torch.full((B, ), prev_timestep, device=self.device, dtype=torch.long)
        else:
            prev_timestep = timestep

        # timestep = torch.full((B, ), timestep, device=self.device, dtype=torch.long)
        # prev_timestep = torch.full((B, ), prev_timestep, device=self.device, dtype=torch.long)

        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep]
        # alpha_prod_t_prev = self.alphas_cumprod_prev[timestep]             ########## wrong
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        c_skip, c_out, c_in = self.get_scalings_for_boundary_condition(timestep)
        # predicted_original_sample = self.sqrt_recip_alphas_cumprod[timestep].reshape(B, *((1, ) * len(x_shape))) * x_t - \
        #     self.sqrt_recipm1_alphas_cumprod[timestep].reshape(B, *((1, ) * len(x_shape))) * noise_pred

        denoised = c_out.reshape(B, *((1, ) * len(x_shape))) * predicted_original_sample + c_skip.reshape(B, *((1, ) * len(x_shape))) * x_t
        # denoised = None
        if step_index != self.num_inference_steps - 1:
            noise = torch.randn_like(x_t)
            # pdb.set_trace()
            model_mean = alpha_prod_t_prev.sqrt().reshape(B, *((1, ) * len(x_shape))) * denoised
            model_std = beta_prod_t_prev.sqrt().reshape(B, *((1, ) * len(x_shape))) # 
            prev_sample = model_mean + model_std * noise
        else:
            model_mean = denoised
            model_std = torch.zeros_like(denoised)
            prev_sample = denoised
        # prev_sample = model_mean + (0.5 * model_log_variance).exp() * noise
        
        return prev_sample, model_mean, model_std

    def p_sample_CM(self, x_t: torch.Tensor, t: int, data: Dict, step_index, timesteps_) -> torch.Tensor:

        B, *_ = x_t.shape
        batch_timestep = torch.full((B, ), t, device=self.device, dtype=torch.long)

        if 'cond' in data:
            ## use precomputed conditional feature
            cond = data['cond']
        else:
            ## recompute conditional feature every sampling step
            cond = self.eps_model.condition(data)
        
        pred_noise, pred_xstart = self.model_predict(x_t, batch_timestep, cond)
        '''
        # if t == 0:
        # pdb.set_trace()
        pred_xstart_ = pred_xstart
        pred_xstart_[:, :3] = self.trans_denormalize(pred_xstart[:, :3])
        pred_xstart_[:, 9:] = self.angle_denormalize(pred_xstart[:, 9:])
        self.visualize_(pred_xstart_, data['scene_id'], t.item(), mark=True)
        # ''' # [:3, :][0:3]
        noise = torch.randn_like(x_t) if step_index == self.num_inference_steps - 1 else 0. # no noise if t == 0

        pred_x, model_mean, model_std = self.step_t_t_1(pred_noise, batch_timestep, x_t, data, timesteps_, predicted_original_sample=pred_xstart, step_index = step_index)

        if self.optimizer is not None:

            return pred_x, pred_xstart, model_mean, model_std

        '''
        if t == 0:
            pred_x_ = pred_x
            pred_x_[:, :3] = self.trans_denormalize(pred_x[:, :3])
            pred_x_[:, 9:] = self.angle_denormalize(pred_x[:, 9:])
            self.visualize_(pred_x_[:1, :], data['scene_id'][:1])
            pdb.set_trace()
        '''
        return pred_x
    
    def p_sample(self, x_t: torch.Tensor, t: int, data: Dict) -> torch.Tensor:
        """ One step of reverse diffusion process

        $x_{t-1} = \tilde{\mu} + \sqrt{\tilde{\beta}} * z$

        Args:
            x_t: denoised sample at timestep t
            t: denoising timestep
            data: data dict that provides original data and computed conditional feature

        Return:
            Predict data in the previous step, i.e., $x_{t-1}$
        """
        B, *_ = x_t.shape
        batch_timestep = torch.full((B, ), t, device=self.device, dtype=torch.long)

        if 'cond' in data:
            ## use precomputed conditional feature
            cond = data['cond']
        else:
            ## recompute conditional feature every sampling step
            cond = self.eps_model.condition(data)
        model_mean, model_variance, model_log_variance, pred_xstart, pred_noise = self.p_mean_variance(x_t, batch_timestep, cond)
        '''
        if t == 0:
            pdb.set_trace()
            pred_xstart_ = pred_xstart
            pred_xstart_[:, :3] = self.trans_denormalize(pred_xstart[:, :3])
            pred_xstart_[:, 9:] = self.angle_denormalize(pred_xstart[:, 9:])
            self.visualize_(pred_xstart_, data['scene_id'])
        '''
        noise = torch.randn_like(x_t) if t > 0 else 0. # no noise if t == 0

        ## sampling with mean updated by optimizer and planner
        # if self.optimizer is not None:
        #     ## openai guided diffusion uses the input x to compute gradient, see
        #     ## https://github.com/openai/guided-diffusion/blob/22e0df8183507e13a7813f8d38d51b072ca1e67c/guided_diffusion/gaussian_diffusion.py#L436
        #     ## But the original formular uses the computed mean?
        #     gradient = self.optimizer.gradient(model_mean, data, model_variance)
        #     model_mean = model_mean + gradient

        pred_x = model_mean + (0.5 * model_log_variance).exp() * noise
        # pdb.set_trace()
        '''
        if t == 0:
            pred_x_ = pred_x
            pred_x_[:, :3] = self.trans_denormalize(pred_x[:, :3])
            pred_x_[:, 9:] = self.angle_denormalize(pred_x[:, 9:])
            self.visualize_(pred_x_[:1, :], data['scene_id'][:1])
            pdb.set_trace()
        '''
        # std_dev_t = self.eta * model_variance.clamp(min=1e-20) ** (0.5)
        # log_prob = (
        # -((pred_x.detach() - model_mean) ** 2) / (2 * (std_dev_t**2))
        # - torch.log(std_dev_t)
        # - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
        # )
        # log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))
        if self.optimizer is not None:
            sample_std = (0.5 * model_log_variance).exp()
            return pred_x, pred_xstart, sample_std , model_mean 
        return pred_x

    def sample_logprob(self, noise_pred, t, latents, prev_sample):
        
        B, *x_shape = latents.shape
        pred_x0 = self.sqrt_recip_alphas_cumprod[t].reshape(B, *((1, ) * len(x_shape))) * latents - \
            self.sqrt_recipm1_alphas_cumprod[t].reshape(B, *((1, ) * len(x_shape))) * noise_pred
        model_mean = self.posterior_mean_coef1[t].reshape(B, *((1, ) * len(x_shape))) * pred_x0 + \
            self.posterior_mean_coef2[t].reshape(B, *((1, ) * len(x_shape))) * latents
        posterior_variance = self.posterior_variance[t].reshape(B, *((1, ) * len(x_shape)))
        # posterior_log_variance = self.posterior_log_variance_clipped[t].reshape(B, *((1, ) * len(x_shape))) # clipped variance
        std_dev_t = self.eta * posterior_variance.clamp(min=1e-5) ** (0.5)

        log_prob = (
        -((latents.detach() - model_mean) ** 2) / (2 * (std_dev_t**2))
        - torch.log(std_dev_t)
        - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
        )
        log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))
        # print(std_dev_t)
        # print(posterior_variance)
        return log_prob

    def sample_logprob_CM(self, noise_pred, timestep, latents, prev_sample):
        B, *x_shape = latents.shape
        # pdb.set_trace()
        step_index = self.index_for_timestep(timestep[0], self.timesteps_)
        prev_step_index = step_index + 1
        if prev_step_index < len(self.timesteps_):
            prev_timestep = self.timesteps_[prev_step_index]
            prev_timestep = torch.full((B, ), prev_timestep, device=self.device, dtype=torch.long)
        else:
            prev_timestep = timestep

        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep]
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        # alpha_prod_t_prev = self.alphas_cumprod_prev[timestep]
        pred_original_sample = self.sqrt_recip_alphas_cumprod[timestep].reshape(B, *((1, ) * len(x_shape))) * latents - \
            self.sqrt_recipm1_alphas_cumprod[timestep].reshape(B, *((1, ) * len(x_shape))) * noise_pred
        c_skip, c_out, c_in = self.get_scalings_for_boundary_condition(timestep)

        denoised = c_out.reshape(B, *((1, ) * len(x_shape))) * pred_original_sample + c_skip.reshape(B, *((1, ) * len(x_shape))) * latents
        # model_mean = alpha_prod_t_prev.sqrt().reshape(B, *((1, ) * len(x_shape))) * denoised
        # std_dev_t = beta_prod_t_prev.sqrt().reshape(B, *((1, ) * len(x_shape))).clamp(min=1e-5)
        # '''
        # pdb.set_trace()
        if step_index != self.num_inference_steps - 1:
            # noise = torch.randn_like(latents)
            model_mean = alpha_prod_t_prev.sqrt().reshape(B, *((1, ) * len(x_shape))) * denoised
            std_dev_t = beta_prod_t_prev.sqrt().reshape(B, *((1, ) * len(x_shape))).clamp(min=1e-5)
            # model_mean = prev_sample - std_dev_t * noise
        else:
            noise = torch.randn_like(latents)
            # model_mean = denoised
            # std_dev_t = torch.zeros_like(denoised).clamp(min=1e-5)
            model_mean = torch.full((B, ), self.alphas_cumprod[0]).to(self.device).sqrt().reshape(B, *((1, ) * len(x_shape))) * denoised
            std_dev_t = torch.full((B, ), 1 - self.alphas_cumprod[0]).to(self.device).sqrt().reshape(B, *((1, ) * len(x_shape))).clamp(min=1e-5)
            # model_mean = alpha_prod_t_prev.sqrt().reshape(B, *((1, ) * len(x_shape))) * denoised
            # std_dev_t = beta_prod_t_prev.sqrt().reshape(B, *((1, ) * len(x_shape))).clamp(min=1e-5)
            # model_mean = prev_sample
            # std_dev_t = ((prev_sample - model_mean) / noise).clamp(min=1e-5)
        # '''
        log_prob = (
        -((latents.detach() - model_mean) ** 2) / (2 * (std_dev_t**2))
        - torch.log(std_dev_t)
        - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
        )
        log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))
        return log_prob

        # variance = get_variance(timestep, prev_timestep)
        # std_dev_t = self.eta * variance.clamp(min=1e-5) ** (0.5)

    def p_sample_loop(self, data: Dict) -> torch.Tensor:
        """ Reverse diffusion process loop, iteratively sampling
        Args:
            data: test data, data['x'] gives the target data shape
        """
        x_t = torch.randn_like(data['x'], device=self.device)
        ## apply observation to x_t
        x_t = self.apply_observation(x_t, data)
        ## precompute conditional feature, which will be used in every sampling step
        with torch.no_grad():
            condition = self.eps_model.condition(data)
        data['cond'] = condition
        ## iteratively sampling
        all_x_t = [x_t]
        all_log_probs = []
        if self.cfg.sample.name == 'dpmsolver++':
            # Initialize the noise schedule for VP-SDE (Variance Preserving Stochastic Differential Equations)
            self.noise_schedule = NoiseScheduleVP(schedule='discrete', betas=self.betas)###dpmsolver++
            # Wrap the noise prediction model in a continuous-time model function
            model_fn_continuous = model_wrapper(
                model=self.eps_model,               # Noise prediction model (epsilon model)
                noise_schedule=self.noise_schedule,     # The noise schedule used for the diffusion process
                model_type="noise",                 # Specify that this is a noise prediction model
                guidance_type="classifier-free",    # Use classifier-free guidance during sampling
                condition=condition,             # Conditional information for the diffusion model
                guidance_scale=1.,                  # Scale of the classifier-free guidance
                classifier_fn=None,                 # No classifier function is used (since this is classifier-free guidance)
                classifier_kwargs={},               # No additional arguments for the classifier
            )

            # Initialize the DPM Solver with the continuous-time model function and noise schedule
            dpm_solver = DPM_Solver(
                model_fn_continuous,                # The continuous-time model function for noise prediction
                self.noise_schedule,
                data = data,                     # The noise schedule for diffusion process
                algorithm_type=self.cfg.sample.name, # Type of DPM Solver algorithm (e.g., dpmsolver++)
                correcting_x0_fn="dynamic_thresholding",              #  function for x0 
                correcting_xt_fn = ((lambda x, x_0_pred, data, x_sample, std: self.optimizer.gradient(x, x_0_pred, data, x_sample=x_sample, std=std)) if self.optimizer is not None else None), # function for xt
            )

            # Generate a sample starting from the noisy input x_t and reconstruct the original data
            x_0 = dpm_solver.sample(
                x_t,                               # Noisy input at time t
                steps=self.cfg.sample.steps,       # Number of sampling steps
                order=self.cfg.sample.order,       # Order of the sampling method (e.g., 1 or 2 or 3 order)
                skip_type=self.cfg.sample.skip_type,    # The type of time step skipping (e.g., 'time_uniform')
                method=self.cfg.sample.method,     # The method for sampling. 'singlestep' or 'multistep' or 'singlestep_fixed' or 'adaptive'.
                t_end=self.cfg.sample.t_end,       # The end time of the diffusion process (usually close to 0)
                t_start=self.cfg.sample.t_start,   # The start time of the diffusion process
            )
            all_x_t.append(x_0)

        elif self.cfg.sample.name == 'DDPM':
            timesteps = []
            if self.optimizer is not None:
                for t in reversed(range(0, self.timesteps)):
                    # pdb.set_trace()
                    if t % self.cfg.opt_interval == 0:
                        ##'TODO:' guidance guradient
                        with torch.enable_grad():
                            x_t = x_t.requires_grad_()
                            x_t_sample, pred_xstart, sample_std, model_mean = self.p_sample(x_t, t, data)
                            x_t = self.optimizer.gradient(x_t, pred_xstart, data, x_mean=model_mean, x_sample = x_t_sample, std=sample_std)
                            x_t = x_t.detach()
                            all_x_t.append(x_t)
                    else:
                        with torch.no_grad(): 
                            x_t, __,__,__,= self.p_sample(x_t, t, data)
                            ## apply observation to x_t
                            x_t = self.apply_observation(x_t, data)
                            all_x_t.append(x_t)
                    timesteps.append(t)
            else:
                with torch.no_grad():
                    for t in reversed(range(0, self.timesteps)):
                        x_t = self.p_sample(x_t, t, data)
                        ## apply observation to x_t
                        x_t = self.apply_observation(x_t, data)
                        
                        all_x_t.append(x_t)
                        # all_log_probs.append(log_prob)
                        timesteps.append(t)
        elif self.cfg.sample.name == 'consistency_model':
            timesteps = []
            if self.optimizer is not None:
                # pdb.set_trace()
                if self.num_inference_steps == 1:
                    timesteps_ = torch.tensor([60], device='cuda:0')
                else:
                    timesteps_ = self.get_sigmas_karras(self.timesteps, self.num_inference_steps, device=self.device)
                self.timesteps_ = timesteps_
                for idx, t in enumerate(timesteps_):
                    with torch.enable_grad():
                        step_index = self.index_for_timestep(t, timesteps_)
                        x_t = x_t.requires_grad_()
                        '''
                        pdb.set_trace()
                        pred_x_ = x_t.detach()
                        pred_x_[:, :3] = self.trans_denormalize(x_t[:, :3])
                        pred_x_[:, 9:] = self.angle_denormalize(x_t[:, 9:])
                        self.visualize_(pred_x_[:16, :], data['scene_id'][:16], t.item(), mark=False)
                        '''
                        x_t_sample, pred_xstart, x_mean, std = self.p_sample_CM(x_t, t, data, step_index, timesteps_)
                        
                        # pred_x_ = x_t.detach()
                        # pred_x_[:, :3] = self.trans_denormalize(x_t[:, :3])
                        # pred_x_[:, 9:] = self.angle_denormalize(x_t[:, 9:])
                        # self.visualize_(pred_x_[:16, :], data['scene_id'][:16], t.item(), mark=True)

                        x_t = x_t.requires_grad_()
                        x_t = self.optimizer.gradient(x_t, pred_xstart, data, x_mean=x_mean, x_sample = x_t_sample, std=std)
                        x_t = self.apply_observation(x_t, data)
                        x_t = x_t.detach()
                        all_x_t.append(x_t)
                        timesteps.append(t)
                # pred_x_ = x_t.detach()
                # pred_x_[:, :3] = self.trans_denormalize(x_t[:, :3])
                # pred_x_[:, 9:] = self.angle_denormalize(x_t[:, 9:])
                # self.visualize_(pred_x_[:1, :], data['scene_id'][:1], 0)
                # pdb.set_trace()
            else:
                with torch.no_grad():
                    timesteps_ = self.get_sigmas_karras(self.timesteps, self.num_inference_steps, device=self.device)
                    self.timesteps_ = timesteps_
                    # for t in reversed(range(0, self.timesteps)):
                    # pdb.set_trace()
                    for idx, t in enumerate(timesteps_):
                        # pdb.set_trace()
                        step_index = self.index_for_timestep(t, timesteps_)
                        x_t = self.p_sample_CM(x_t, t, data, step_index, timesteps_)

                        x_t = self.apply_observation(x_t, data)
                        
                        all_x_t.append(x_t)
                        timesteps.append(t)
        elif self.cfg.sample.name == 'short_cut':
            if self.optimizer is not None:
                raise ValueError("this part has not been developed")
            else:
                # pdb.set_trace()
                denoise_timesteps = 1       ## [1, 2, 4, 8, 16, 32, 128]
                delta_t = 1.0 / denoise_timesteps
                with torch.no_grad():
                    timesteps = []
                    for ti in range(denoise_timesteps):
                        t = ti / denoise_timesteps
                        t_vector = torch.full((x_t.shape[0],), t).to(self.device)

                        dt_base = torch.ones_like(t_vector).to(self.device) * delta_t # math.log2(denoise_timesteps)

                        v = self.eps_model(x_t, t_vector, data['cond'], dt_base)
                        x_t = x_t + v * delta_t
                        all_x_t.append(x_t)
                        timesteps.append(t)

        return torch.stack(all_x_t, dim=0), None, torch.tensor(timesteps, device = self.device) # torch.stack(all_log_probs, dim=1)
    
    def index_for_timestep(self, timestep, schedule_timesteps=None):
        if schedule_timesteps is None:
            schedule_timesteps = self.timesteps

        indices = (schedule_timesteps == timestep).nonzero()

        pos = 1 if len(indices) > 1 else 0

        return indices[pos].item()

    def sample(self, data: Dict, k: int=1) -> torch.Tensor:
        """ Reverse diffusion process, sampling with the given data containing condition
        In this method, the sampled results are unnormalized and converted to absolute representation.

        Args:
            data: test data, data['x'] gives the target data shape
            k: the number of sampled data
        
        Return:
            Sampled results, the shape is <B, k, T, ...>
        """
        ## TODO ddim sample function
        ksamples = []
        # log_probs = []
        timesteps = []
        for _ in range(k):
            x_t, _, timestep = self.p_sample_loop(data)
            ksamples.append(x_t)
            # log_probs.append(log_prob)
            timesteps.append(timestep)
        
        ksamples = torch.stack(ksamples, dim = 1)
        # log_probs = torch.stack(log_probs, dim = 1)
        timesteps = torch.stack(timesteps, dim = 0)

        ## for sequence, normalize and convert repr
        if 'normalizer' in data and data['normalizer'] is not None:
            O = 0
            if self.has_observation and 'start' in data:
                ## the start observation frames are replace during sampling
                _, O, _ = data['start'].shape
            ksamples[..., O:, :] = data['normalizer'].unnormalize(ksamples[..., O:, :])
        if 'repr_type' in data:
            if data['repr_type'] == 'absolute':
                pass
            elif data['repr_type'] == 'relative':
                O = 1
                if self.has_observation and 'start' in data:
                    _, O, _ = data['start'].shape
                ksamples[..., O-1:, :] = torch.cumsum(ksamples[..., O-1:, :], dim=-2)
            else:
                raise Exception('Unsupported repr type.')
        
        return ksamples, None, timesteps
    
    def set_optimizer(self, optimizer: Optimizer):
        """ Set optimizer for diffuser, the optimizer is used in sampling

        Args:
            optimizer: a Optimizer object that has a gradient method
        """
        self.optimizer = optimizer
    
    def set_planner(self, planner: Planner):
        """ Set planner for diffuser, the planner is used in sampling

        Args:
            planner: a Planner object that has a gradient method
        """
        self.planner = planner
    
    def DPO_forward(self, data: Dict, ref: torch.nn.Module, cfg):
        # pdb.set_trace()
        B, *x_shape = data['x'].shape
        ## randomly sample timesteps

        self.timesteps_ = self.get_sigmas_karras(self.timesteps, self.num_inference_steps, device=self.device)

        random_indices = torch.randint(0, len(self.timesteps_), (B,), device=self.device)
        ts = self.timesteps_[random_indices]
        # if self.rand_t_type == 'all':
        #     ts = torch.randint(0, self.timesteps, (B, ), device=self.device).long()
        # elif self.rand_t_type == 'half':
        #     ts = torch.randint(0, self.timesteps, ((B + 1) // 2, ), device=self.device)
        #     if B % 2 == 1:
        #         ts = torch.cat([ts, self.timesteps - ts[:-1] - 1], dim=0).long()
        #     else:
        #         ts = torch.cat([ts, self.timesteps - ts - 1], dim=0).long()
        # else:
        #     raise Exception('Unsupported rand ts type.')
        
        ## generate Gaussian noise
        noise = torch.randn_like(data['x'], device=self.device)

        ## calculate x_t, forward diffusion process
        x_t = self.q_sample(x0=data['x'], t=ts, noise=noise)
        # x_t_neg = self.q_sample(x0=data['x_neg'], t=ts, noise=noise)
        ## apply observation before forwarding to eps model
        ## model need to learn the relationship between the observation frames and the rest frames
        x_t = self.apply_observation(x_t, data)
        # x_t_neg = self.apply_observation(x_t_neg, data)
        # x_t = torch.cat((x_t, x_t_neg), dim=0)
        
        condtion = self.eps_model.condition(data)
        output = self.eps_model(x_t, ts, condtion)
        # output = self.eps_model(x_t, torch.cat((ts, ts), dim=0), torch.cat((condtion, condtion), dim=0))
        ref_output = ref(x_t, ts, condtion)
        # ref_output = ref(x_t, torch.cat((ts, ts), dim=0), torch.cat((condtion, condtion), dim=0))
        ## apply observation after forwarding to eps model
        ## this operation will detach gradient from the loss of the observation tokens
        ## because the target and output of the observation tokens all are constants
        output = self.apply_observation(output, data)
        ref_output = self.apply_observation(ref_output, data)
        # ''''''
        model_loss = (output - noise).pow(2)
        ref_loss = (ref_output - noise).pow(2)
        pdb.set_trace()
        step = 0
        diff_pos = 0
        diff_neg = 0
        for idx in data['label']:
            if idx == 1:
                diff_pos = diff_pos + model_loss[step, :] - ref_loss[step, :]
            else:
                diff_neg = diff_neg + model_loss[step, :] - ref_loss[step, :]
            step = step + 1
        # diff_pos = model_loss[:B, :] - ref_loss[:B, :]
        # diff_neg = model_loss[B:, :] - ref_loss[B:, :]
        # log_prob = self.sample_logprob_CM(output, torch.cat((ts, ts), dim=0), x_t, None)
        # ref_prob = self.sample_logprob_CM(ref_output, torch.cat((ts, ts), dim=0), x_t, None)
        inside_term = -cfg.beta * (diff_pos - diff_neg)#.clamp(min=1e-5)
        # for i in range(len(total_ref_prob_0)):
        #     ratio[i] = torch.clamp(torch.exp(total_prob_0[i]-total_ref_prob_0[i]),1 - eps, 1 + eps)
        #     tmp = tmp + cfg.beta*(torch.log(ratio[i]))*rewards[i]
        # pdb.set_trace()
        # tmp = sum(tmp_)
        loss = -torch.log(torch.sigmoid(inside_term)).mean()                     ################### + or -

        print(f'loss: {loss}')
        return {'loss': loss}
        
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

    def short_cut_forward(self, data: Dict) -> torch.Tensor:
        condtion = self.eps_model.condition(data)
        
        x_t, v_t, t, dt_base, labels_dropped = self.create_targets(data['x'], condtion)

        v_prime = self.eps_model(x_t, t, condtion, dt_base)
        loss = self.criterion(v_prime, v_t)

        return {'loss': loss}
    
    # create batch, consisting of different timesteps and different dts(depending on total step sizes)
    def create_targets(self, qpos, condtion):

        self.eps_model.eval()

        current_batch_size = qpos.shape[0]
        pdb.set_trace()
        FORCE_T = -1
        FORCE_DT = -1

        # 1. create step sizes dt
        bootstrap_batch_size = current_batch_size // self.BOOTSTRAP_EVERY #=8
        log2_sections = int(math.log2(self.DENOISE_TIMESTEPS))
        # print(f"log2_sections: {log2_sections}")
        # print(f"bootstrap_batch_size: {bootstrap_batch_size}")

        dt_base = torch.repeat_interleave(log2_sections - 1 - torch.arange(log2_sections), bootstrap_batch_size // log2_sections)
        # print(f"dt_base: {dt_base}")

        dt_base = torch.cat([dt_base, torch.zeros(bootstrap_batch_size-dt_base.shape[0],)])
        # print(f"dt_base: {dt_base}")

        force_dt_vec = torch.ones(bootstrap_batch_size) * FORCE_DT
        dt_base = torch.where(force_dt_vec != -1, force_dt_vec, dt_base).to(self.device)
        dt = 1 / (2 ** (dt_base)) # [1, 1/2, 1/8, 1/16, 1/32]
        # print(f"dt: {dt}")

        dt_base_bootstrap = dt_base + 1
        dt_bootstrap = dt / 2 # [0.0078125 0.015625 0.03125 0.0625 0.125 0.25 0.5 0.5]
        # print(f"dt_bootstrap: {dt_bootstrap}")

        # 2. sample timesteps t
        dt_sections = 2**dt_base

        # print(f"dt_sections: {dt_sections}")

        t = torch.cat([
            torch.randint(low=0, high=int(val.item()), size=(1,)).float()
            for val in dt_sections
            ]).to(self.device)
        
        # print(f"t[randint]: {t}")
        t = t / dt_sections
        # print(f"t[normalized]: {t}")
        
        force_t_vec = torch.ones(bootstrap_batch_size, dtype=torch.float32).to(self.device) * FORCE_T
        t = torch.where(force_t_vec != -1, force_t_vec, t).to(self.device)
        t_full = t[:, None]

        # print(f"t_full: {t_full}")

        # 3. generate bootstrap targets:
        x_1 = qpos[:bootstrap_batch_size]
        x_0 = torch.randn_like(x_1)

        # get dx at timestep t
        x_t = (1 - (1-1e-5) * t_full)*x_0 + t_full*x_1

        bst_condtion = condtion[:bootstrap_batch_size]
        
        with torch.no_grad():
            v_b1 = self.eps_model(x_t, t, bst_condtion, dt_base_bootstrap)
        
        t2 = t + dt_bootstrap
        x_t2 = x_t + dt_bootstrap[:, None] * v_b1
        x_t2 = torch.clip(x_t2, -4, 4)
        
        with torch.no_grad():
            v_b2 = self.eps_model(x_t2, t2, bst_condtion, dt_base_bootstrap)

        v_target = (v_b1 + v_b2) / 2

        v_target = torch.clip(v_target, -4, 4)
        
        bst_v = v_target
        bst_dt = dt_base
        bst_t = t
        bst_xt = x_t
        bst_cond = bst_condtion

        # 4. generate flow-matching targets
        # pdb.set_trace()
        # labels_dropout = torch.bernoulli(torch.full(condtion.shape, self.CLASS_DROPOUT_PROB)).to(self.device)
        # labels_dropped = torch.where(labels_dropout.bool(), self.NUM_CLASSES, condtion)

        # sample t(normalized)
        t = torch.randint(low=0, high=self.DENOISE_TIMESTEPS, size=(qpos.shape[0],), dtype=torch.float32)
        # print(f"t: {t}")
        t /= self.DENOISE_TIMESTEPS
        # print(f"t: {t}")
        force_t_vec = torch.ones(qpos.shape[0]) * FORCE_T
        # force_t_vec = torch.full((qpos.shape[0],), FORCE_T, dtype=torch.float32)
        t = torch.where(force_t_vec != -1, force_t_vec, t).to(self.device)
        # t_full = t.view(-1, 1, 1, 1)
        t_full = t[:, None]

        # print(f"t_full: {t_full}")

        # sample flow pairs x_t, v_t
        x_0 = torch.randn_like(qpos).to(self.device)
        x_1 = qpos
        x_t = (1 - (1 - 1e-5) * t_full) * x_0 + t_full * x_1
        v_t = x_1 - (1 - 1e-5) * x_0
        # pdb.set_trace()
        dt_flow = int(math.log2(self.DENOISE_TIMESTEPS))
        # dt_base = (torch.ones(qpos.shape[0], dtype=torch.int32) * dt_flow).to(self.device)
        dt_base = (torch.zeros(qpos.shape[0], dtype=torch.int32) * dt_flow).to(self.device)

        # 5. merge flow and bootstrap
        bst_size = current_batch_size // self.BOOTSTRAP_EVERY
        bst_size_data = current_batch_size - bst_size

        # print(f"bst_size: {bst_size}")
        # print(f"bst_size_data: {bst_size_data}")

        x_t = torch.cat([bst_xt, x_t[:bst_size_data]], dim=0)
        t = torch.cat([bst_t, t[:bst_size_data]], dim=0)

        dt_base = torch.cat([bst_dt, dt_base[:bst_size_data]], dim=0)
        v_t = torch.cat([bst_v, v_t[:bst_size_data]], dim=0)
        # labels_dropped = torch.cat([bst_cond, labels_dropped[:bst_size_data]], dim=0)

        return x_t, v_t, t, dt_base, None

    def save_point_cloud_with_normals(self, hand_pcd: torch.Tensor, obj_pcd: torch.Tensor, obj_normals: torch.Tensor):
        """
        Save point cloud data with normals for visualization.

        Args:
            hand_pcd: B x N_hand x 3 tensor of hand point cloud
            obj_pcd: B x N_obj x 3 tensor of object point cloud
            obj_normals: B x N_obj x 3 tensor of object normals
        """
        B = hand_pcd.shape[0]
        for i in range(B):
            hand_pcd_np = hand_pcd[i].detach().cpu().numpy()
            obj_pcd_np = obj_pcd[i].detach().cpu().numpy()
            obj_normals_np = obj_normals[i].detach().cpu().numpy()

            # Combine object points and normals
            obj_pcd_with_normals = np.hstack((obj_pcd_np, obj_normals_np))

            # Save object point cloud with normals
            vertex = np.array([tuple(row) for row in obj_pcd_with_normals],
                            dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                                    ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4')])
            el = PlyElement.describe(vertex, 'vertex')
            PlyData([el]).write(f'/inspurfs/group/mayuexin/zhuyufei/graps_gen/outputs/vis_point_cloud_obj/obj_pcd_with_normals_{i}.ply')

            # Save hand point cloud
            vertex_hand = np.array([tuple(row) for row in hand_pcd_np],
                                dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
            el_hand = PlyElement.describe(vertex_hand, 'vertex')
            PlyData([el_hand]).write(f'/inspurfs/group/mayuexin/zhuyufei/graps_gen/outputs/vis_point_cloud_hand/hand_pcd_{i}.ply')

#############################################################
# dpm_solver++
#############################################################

class NoiseScheduleVP:
    def __init__(
            self,
            schedule='discrete',
            betas=None,
            alphas_cumprod=None,
            continuous_beta_0=0.1,
            continuous_beta_1=20.,
            dtype=torch.float32,
        ):
        """Create a wrapper class for the forward SDE (VP type).

        ***
        Update: We support discrete-time diffusion models by implementing a picewise linear interpolation for log_alpha_t.
                We recommend to use schedule='discrete' for the discrete-time diffusion models, especially for high-resolution images.
        ***

        The forward SDE ensures that the condition distribution q_{t|0}(x_t | x_0) = N ( alpha_t * x_0, sigma_t^2 * I ).
        We further define lambda_t = log(alpha_t) - log(sigma_t), which is the half-logSNR (described in the DPM-Solver paper).
        Therefore, we implement the functions for computing alpha_t, sigma_t and lambda_t. For t in [0, T], we have:

            log_alpha_t = self.marginal_log_mean_coeff(t)
            sigma_t = self.marginal_std(t)
            lambda_t = self.marginal_lambda(t)

        Moreover, as lambda(t) is an invertible function, we also support its inverse function:

            t = self.inverse_lambda(lambda_t)

        ===============================================================

        We support both discrete-time DPMs (trained on n = 0, 1, ..., N-1) and continuous-time DPMs (trained on t in [t_0, T]).

        1. For discrete-time DPMs:

            For discrete-time DPMs trained on n = 0, 1, ..., N-1, we convert the discrete steps to continuous time steps by:
                t_i = (i + 1) / N
            e.g. for N = 1000, we have t_0 = 1e-3 and T = t_{N-1} = 1.
            We solve the corresponding diffusion ODE from time T = 1 to time t_0 = 1e-3.

            Args:
                betas: A `torch.Tensor`. The beta array for the discrete-time DPM. (See the original DDPM paper for details)
                alphas_cumprod: A `torch.Tensor`. The cumprod alphas for the discrete-time DPM. (See the original DDPM paper for details)

            Note that we always have alphas_cumprod = cumprod(1 - betas). Therefore, we only need to set one of `betas` and `alphas_cumprod`.

            **Important**:  Please pay special attention for the args for `alphas_cumprod`:
                The `alphas_cumprod` is the \hat{alpha_n} arrays in the notations of DDPM. Specifically, DDPMs assume that
                    q_{t_n | 0}(x_{t_n} | x_0) = N ( \sqrt{\hat{alpha_n}} * x_0, (1 - \hat{alpha_n}) * I ).
                Therefore, the notation \hat{alpha_n} is different from the notation alpha_t in DPM-Solver. In fact, we have
                    alpha_{t_n} = \sqrt{\hat{alpha_n}},
                and
                    log(alpha_{t_n}) = 0.5 * log(\hat{alpha_n}).


        2. For continuous-time DPMs:

            We support the linear VPSDE for the continuous time setting. The hyperparameters for the noise
            schedule are the default settings in Yang Song's ScoreSDE:

            Args:
                beta_min: A `float` number. The smallest beta for the linear schedule.
                beta_max: A `float` number. The largest beta for the linear schedule.
                T: A `float` number. The ending time of the forward process.

        ===============================================================

        Args:
            schedule: A `str`. The noise schedule of the forward SDE. 'discrete' for discrete-time DPMs,
                    'linear' for continuous-time DPMs.
        Returns:
            A wrapper object of the forward SDE (VP type).
        
        ===============================================================

        Example:

        # For discrete-time DPMs, given betas (the beta array for n = 0, 1, ..., N - 1):
        >>> ns = NoiseScheduleVP('discrete', betas=betas)

        # For discrete-time DPMs, given alphas_cumprod (the \hat{alpha_n} array for n = 0, 1, ..., N - 1):
        >>> ns = NoiseScheduleVP('discrete', alphas_cumprod=alphas_cumprod)

        # For continuous-time DPMs (VPSDE), linear schedule:
        >>> ns = NoiseScheduleVP('linear', continuous_beta_0=0.1, continuous_beta_1=20.)

        """

        if schedule not in ['discrete', 'linear']:
            raise ValueError("Unsupported noise schedule {}. The schedule needs to be 'discrete' or 'linear'".format(schedule))

        self.schedule = schedule
        if schedule == 'discrete':
            if betas is not None:
                log_alphas = 0.5 * torch.log(1 - betas).cumsum(dim=0)
            else:
                assert alphas_cumprod is not None
                log_alphas = 0.5 * torch.log(alphas_cumprod)
            self.T = 1.
            self.log_alpha_array = self.numerical_clip_alpha(log_alphas).reshape((1, -1,)).to(dtype=dtype)
            self.total_N = self.log_alpha_array.shape[1]
            self.t_array = torch.linspace(0., 1., self.total_N + 1)[1:].reshape((1, -1)).to(dtype=dtype)
        else:
            self.T = 1.
            self.total_N = 1000
            self.beta_0 = continuous_beta_0
            self.beta_1 = continuous_beta_1

    def numerical_clip_alpha(self, log_alphas, clipped_lambda=-5.1):
        """
        For some beta schedules such as cosine schedule, the log-SNR has numerical isssues. 
        We clip the log-SNR near t=T within -5.1 to ensure the stability.
        Such a trick is very useful for diffusion models with the cosine schedule, such as i-DDPM, guided-diffusion and GLIDE.
        """
        log_sigmas = 0.5 * torch.log(1. - torch.exp(2. * log_alphas))
        lambs = log_alphas - log_sigmas  
        idx = torch.searchsorted(torch.flip(lambs, [0]), clipped_lambda)
        if idx > 0:
            log_alphas = log_alphas[:-idx]
        return log_alphas

    def marginal_log_mean_coeff(self, t):
        """
        Compute log(alpha_t) of a given continuous-time label t in [0, T].
        """
        if self.schedule == 'discrete':
            return interpolate_fn(t.reshape((-1, 1)), self.t_array.to(t.device), self.log_alpha_array.to(t.device)).reshape((-1))
        elif self.schedule == 'linear':
            return -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0

    def marginal_alpha(self, t):
        """
        Compute alpha_t of a given continuous-time label t in [0, T].
        """
        return torch.exp(self.marginal_log_mean_coeff(t))

    def marginal_std(self, t):
        """
        Compute sigma_t of a given continuous-time label t in [0, T].
        """
        return torch.sqrt(1. - torch.exp(2. * self.marginal_log_mean_coeff(t)))

    def marginal_lambda(self, t):
        """
        Compute lambda_t = log(alpha_t) - log(sigma_t) of a given continuous-time label t in [0, T].
        """
        log_mean_coeff = self.marginal_log_mean_coeff(t)
        log_std = 0.5 * torch.log(1. - torch.exp(2. * log_mean_coeff))
        return log_mean_coeff - log_std

    def inverse_lambda(self, lamb):
        """
        Compute the continuous-time label t in [0, T] of a given half-logSNR lambda_t.
        """
        if self.schedule == 'linear':
            tmp = 2. * (self.beta_1 - self.beta_0) * torch.logaddexp(-2. * lamb, torch.zeros((1,)).to(lamb))
            Delta = self.beta_0**2 + tmp
            return tmp / (torch.sqrt(Delta) + self.beta_0) / (self.beta_1 - self.beta_0)
        elif self.schedule == 'discrete':
            log_alpha = -0.5 * torch.logaddexp(torch.zeros((1,)).to(lamb.device), -2. * lamb)
            t = interpolate_fn(log_alpha.reshape((-1, 1)), torch.flip(self.log_alpha_array.to(lamb.device), [1]), torch.flip(self.t_array.to(lamb.device), [1]))
            return t.reshape((-1,))


def model_wrapper(
    model,
    noise_schedule,
    model_type="noise",
    model_kwargs={},
    guidance_type="uncond",
    condition=None,
    unconditional_condition=None,
    guidance_scale=1.,
    classifier_fn=None,
    classifier_kwargs={},
):
    """Create a wrapper function for the noise prediction model.

    DPM-Solver needs to solve the continuous-time diffusion ODEs. For DPMs trained on discrete-time labels, we need to
    firstly wrap the model function to a noise prediction model that accepts the continuous time as the input.

    We support four types of the diffusion model by setting `model_type`:

        1. "noise": noise prediction model. (Trained by predicting noise).

        2. "x_start": data prediction model. (Trained by predicting the data x_0 at time 0).

        3. "v": velocity prediction model. (Trained by predicting the velocity).
            The "v" prediction is derivation detailed in Appendix D of [1], and is used in Imagen-Video [2].

            [1] Salimans, Tim, and Jonathan Ho. "Progressive distillation for fast sampling of diffusion models."
                arXiv preprint arXiv:2202.00512 (2022).
            [2] Ho, Jonathan, et al. "Imagen Video: High Definition Video Generation with Diffusion Models."
                arXiv preprint arXiv:2210.02303 (2022).
    
        4. "score": marginal score function. (Trained by denoising score matching).
            Note that the score function and the noise prediction model follows a simple relationship:
            ```
                noise(x_t, t) = -sigma_t * score(x_t, t)
            ```

    We support three types of guided sampling by DPMs by setting `guidance_type`:
        1. "uncond": unconditional sampling by DPMs.
            The input `model` has the following format:
            ``
                model(x, t_input, **model_kwargs) -> noise | x_start | v | score
            ``

        2. "classifier": classifier guidance sampling [3] by DPMs and another classifier.
            The input `model` has the following format:
            ``
                model(x, t_input, **model_kwargs) -> noise | x_start | v | score
            `` 

            The input `classifier_fn` has the following format:
            ``
                classifier_fn(x, t_input, cond, **classifier_kwargs) -> logits(x, t_input, cond)
            ``

            [3] P. Dhariwal and A. Q. Nichol, "Diffusion models beat GANs on image synthesis,"
                in Advances in Neural Information Processing Systems, vol. 34, 2021, pp. 8780-8794.

        3. "classifier-free": classifier-free guidance sampling by conditional DPMs.
            The input `model` has the following format:
            ``
                model(x, t_input, cond, **model_kwargs) -> noise | x_start | v | score
            `` 
            And if cond == `unconditional_condition`, the model output is the unconditional DPM output.

            [4] Ho, Jonathan, and Tim Salimans. "Classifier-free diffusion guidance."
                arXiv preprint arXiv:2207.12598 (2022).
        

    The `t_input` is the time label of the model, which may be discrete-time labels (i.e. 0 to 999)
    or continuous-time labels (i.e. epsilon to T).

    We wrap the model function to accept only `x` and `t_continuous` as inputs, and outputs the predicted noise:
    ``
        def model_fn(x, t_continuous) -> noise:
            t_input = get_model_input_time(t_continuous)
            return noise_pred(model, x, t_input, **model_kwargs)         
    ``
    where `t_continuous` is the continuous time labels (i.e. epsilon to T). And we use `model_fn` for DPM-Solver.

    ===============================================================

    Args:
        model: A diffusion model with the corresponding format described above.
        noise_schedule: A noise schedule object, such as NoiseScheduleVP.
        model_type: A `str`. The parameterization type of the diffusion model.
                    "noise" or "x_start" or "v" or "score".
        model_kwargs: A `dict`. A dict for the other inputs of the model function.
        guidance_type: A `str`. The type of the guidance for sampling.
                    "uncond" or "classifier" or "classifier-free".
        condition: A pytorch tensor. The condition for the guided sampling.
                    Only used for "classifier" or "classifier-free" guidance type.
        unconditional_condition: A pytorch tensor. The condition for the unconditional sampling.
                    Only used for "classifier-free" guidance type.
        guidance_scale: A `float`. The scale for the guided sampling.
        classifier_fn: A classifier function. Only used for the classifier guidance.
        classifier_kwargs: A `dict`. A dict for the other inputs of the classifier function.
    Returns:
        A noise prediction model that accepts the noised data and the continuous time as the inputs.
    """

    def get_model_input_time(t_continuous):
        """
        Convert the continuous-time `t_continuous` (in [epsilon, T]) to the model input time.
        For discrete-time DPMs, we convert `t_continuous` in [1 / N, 1] to `t_input` in [0, 1000 * (N - 1) / N].
        For continuous-time DPMs, we just use `t_continuous`.
        """
        if noise_schedule.schedule == 'discrete':
            return (t_continuous - 1. / noise_schedule.total_N) * noise_schedule.total_N
        else:
            return t_continuous

    def noise_pred_fn(x, t_continuous, cond=None):
        t_input = get_model_input_time(t_continuous)
        if cond is None:
            output = model(x, t_input, **model_kwargs)
        else:
            output = model(x, t_input, cond, **model_kwargs)
        if model_type == "noise":
            return output
        elif model_type == "x_start":
            alpha_t, sigma_t = noise_schedule.marginal_alpha(t_continuous), noise_schedule.marginal_std(t_continuous)
            return (x - alpha_t * output) / sigma_t
        elif model_type == "v":
            alpha_t, sigma_t = noise_schedule.marginal_alpha(t_continuous), noise_schedule.marginal_std(t_continuous)
            return alpha_t * output + sigma_t * x
        elif model_type == "score":
            sigma_t = noise_schedule.marginal_std(t_continuous)
            return -sigma_t * output

    def cond_grad_fn(x, t_input):
        """
        Compute the gradient of the classifier, i.e. nabla_{x} log p_t(cond | x_t).
        """
        with torch.enable_grad():
            x_in = x.detach().requires_grad_(True)
            log_prob = classifier_fn(x_in, t_input, condition, **classifier_kwargs)
            return torch.autograd.grad(log_prob.sum(), x_in)[0]

    def model_fn(x, t_continuous):
        """
        The noise predicition model function that is used for DPM-Solver.
        """
        if guidance_type == "uncond":
            return noise_pred_fn(x, t_continuous)
        elif guidance_type == "classifier":
            assert classifier_fn is not None
            t_input = get_model_input_time(t_continuous)
            cond_grad = cond_grad_fn(x, t_input)
            sigma_t = noise_schedule.marginal_std(t_continuous)
            noise = noise_pred_fn(x, t_continuous)
            return noise - guidance_scale * expand_dims(sigma_t, x.dim()) * cond_grad
        elif guidance_type == "classifier-free":
            if guidance_scale == 1. or unconditional_condition is None:
                return noise_pred_fn(x, t_continuous, cond=condition)
            else:
                x_in = torch.cat([x] * 2)
                t_in = torch.cat([t_continuous] * 2)
                c_in = torch.cat([unconditional_condition, condition])
                noise_uncond, noise = noise_pred_fn(x_in, t_in, cond=c_in).chunk(2)
                return noise_uncond + guidance_scale * (noise - noise_uncond)

    assert model_type in ["noise", "x_start", "v", "score"]
    assert guidance_type in ["uncond", "classifier", "classifier-free"]
    return model_fn


class DPM_Solver:
    def __init__(
        self,
        model_fn,
        noise_schedule,
        data = None,
        algorithm_type="dpmsolver++",
        correcting_x0_fn=None,
        correcting_xt_fn=None,
        thresholding_max_val=1.,
        dynamic_thresholding_ratio=0.995,
    ):
        """Construct a DPM-Solver. 

        We support both DPM-Solver (`algorithm_type="dpmsolver"`) and DPM-Solver++ (`algorithm_type="dpmsolver++"`).

        We also support the "dynamic thresholding" method in Imagen[1]. For pixel-space diffusion models, you
        can set both `algorithm_type="dpmsolver++"` and `correcting_x0_fn="dynamic_thresholding"` to use the
        dynamic thresholding. The "dynamic thresholding" can greatly improve the sample quality for pixel-space
        DPMs with large guidance scales. Note that the thresholding method is **unsuitable** for latent-space
        DPMs (such as stable-diffusion).

        To support advanced algorithms in image-to-image applications, we also support corrector functions for
        both x0 and xt.

        Args:
            model_fn: A noise prediction model function which accepts the continuous-time input (t in [epsilon, T]):
                ``
                def model_fn(x, t_continuous):
                    return noise
                ``
                The shape of `x` is `(batch_size, **shape)`, and the shape of `t_continuous` is `(batch_size,)`.
            noise_schedule: A noise schedule object, such as NoiseScheduleVP.
            algorithm_type: A `str`. Either "dpmsolver" or "dpmsolver++".
            correcting_x0_fn: A `str` or a function with the following format:
                ```
                def correcting_x0_fn(x0, t):
                    x0_new = ...
                    return x0_new
                ```
                This function is to correct the outputs of the data prediction model at each sampling step. e.g.,
                ```
                x0_pred = data_pred_model(xt, t)
                if correcting_x0_fn is not None:
                    x0_pred = correcting_x0_fn(x0_pred, t)
                xt_1 = update(x0_pred, xt, t)
                ```
                If `correcting_x0_fn="dynamic_thresholding"`, we use the dynamic thresholding proposed in Imagen[1].
            correcting_xt_fn: A function with the following format:
                ```
                def correcting_xt_fn(xt, t, step):
                    x_new = ...
                    return x_new
                ```
                This function is to correct the intermediate samples xt at each sampling step. e.g.,
                ```
                xt = ...
                xt = correcting_xt_fn(xt, t, step)
                ```
            thresholding_max_val: A `float`. The max value for thresholding.
                Valid only when use `dpmsolver++` and `correcting_x0_fn="dynamic_thresholding"`.
            dynamic_thresholding_ratio: A `float`. The ratio for dynamic thresholding (see Imagen[1] for details).
                Valid only when use `dpmsolver++` and `correcting_x0_fn="dynamic_thresholding"`.

        [1] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily Denton, Seyed Kamyar Seyed Ghasemipour,
            Burcu Karagol Ayan, S Sara Mahdavi, Rapha Gontijo Lopes, et al. Photorealistic text-to-image diffusion models
            with deep language understanding. arXiv preprint arXiv:2205.11487, 2022b.
        """
        self.model = lambda x, t: model_fn(x, t.expand((x.shape[0])))
        self.noise_schedule = noise_schedule
        assert algorithm_type in ["dpmsolver", "dpmsolver++"]
        self.algorithm_type = algorithm_type
        if correcting_x0_fn == "dynamic_thresholding":
            self.correcting_x0_fn = self.dynamic_thresholding_fn
        else:
            self.correcting_x0_fn = correcting_x0_fn
        self.correcting_xt_fn = correcting_xt_fn
        self.dynamic_thresholding_ratio = dynamic_thresholding_ratio
        self.thresholding_max_val = thresholding_max_val
        self.data = data
    def dynamic_thresholding_fn(self, x0, t):
        """
        The dynamic thresholding method. 
        """
        dims = x0.dim()
        p = self.dynamic_thresholding_ratio
        s = torch.quantile(torch.abs(x0).reshape((x0.shape[0], -1)), p, dim=1)
        s = expand_dims(torch.maximum(s, self.thresholding_max_val * torch.ones_like(s).to(s.device)), dims)
        x0 = torch.clamp(x0, -s, s) / s
        return x0
    
    def noise_prediction_fn(self, x, t):
        """
        Return the noise prediction model.
        """
        return self.model(x, t)

    def data_prediction_fn(self, x, t):
        """
        Return the data prediction model (with corrector).
        """

        noise = self.noise_prediction_fn(x, t)
        alpha_t, sigma_t = self.noise_schedule.marginal_alpha(t), self.noise_schedule.marginal_std(t)
        x0 = (x - sigma_t * noise) / alpha_t
        if self.correcting_x0_fn is not None:
            x0 = self.correcting_x0_fn(x0, t)

        return x0
    
    def model_fn(self, x, t):
        """
        Convert the model to the noise prediction model or the data prediction model. 
        """
        if self.algorithm_type == "dpmsolver++":
            return self.data_prediction_fn(x, t)
        else:
            return self.noise_prediction_fn(x, t)

    def get_time_steps(self, skip_type, t_T, t_0, N, device):
        """Compute the intermediate time steps for sampling.

        Args:
            skip_type: A `str`. The type for the spacing of the time steps. We support three types:
                - 'logSNR': uniform logSNR for the time steps.
                - 'time_uniform': uniform time for the time steps. (**Recommended for high-resolutional data**.)
                - 'time_quadratic': quadratic time for the time steps. (Used in DDIM for low-resolutional data.)
            t_T: A `float`. The starting time of the sampling (default is T).
            t_0: A `float`. The ending time of the sampling (default is epsilon).
            N: A `int`. The total number of the spacing of the time steps.
            device: A torch device.
        Returns:
            A pytorch tensor of the time steps, with the shape (N + 1,).
        """
        if skip_type == 'logSNR':
            lambda_T = self.noise_schedule.marginal_lambda(torch.tensor(t_T).to(device))
            lambda_0 = self.noise_schedule.marginal_lambda(torch.tensor(t_0).to(device))
            logSNR_steps = torch.linspace(lambda_T.cpu().item(), lambda_0.cpu().item(), N + 1).to(device)
            return self.noise_schedule.inverse_lambda(logSNR_steps)
        elif skip_type == 'time_uniform':
            return torch.linspace(t_T, t_0, N + 1).to(device)
        elif skip_type == 'time_quadratic':
            t_order = 2
            t = torch.linspace(t_T**(1. / t_order), t_0**(1. / t_order), N + 1).pow(t_order).to(device)
            return t
        else:
            raise ValueError("Unsupported skip_type {}, need to be 'logSNR' or 'time_uniform' or 'time_quadratic'".format(skip_type))

    def get_orders_and_timesteps_for_singlestep_solver(self, steps, order, skip_type, t_T, t_0, device):
        """
        Get the order of each step for sampling by the singlestep DPM-Solver.

        We combine both DPM-Solver-1,2,3 to use all the function evaluations, which is named as "DPM-Solver-fast".
        Given a fixed number of function evaluations by `steps`, the sampling procedure by DPM-Solver-fast is:
            - If order == 1:
                We take `steps` of DPM-Solver-1 (i.e. DDIM).
            - If order == 2:
                - Denote K = (steps // 2). We take K or (K + 1) intermediate time steps for sampling.
                - If steps % 2 == 0, we use K steps of DPM-Solver-2.
                - If steps % 2 == 1, we use K steps of DPM-Solver-2 and 1 step of DPM-Solver-1.
            - If order == 3:
                - Denote K = (steps // 3 + 1). We take K intermediate time steps for sampling.
                - If steps % 3 == 0, we use (K - 2) steps of DPM-Solver-3, and 1 step of DPM-Solver-2 and 1 step of DPM-Solver-1.
                - If steps % 3 == 1, we use (K - 1) steps of DPM-Solver-3 and 1 step of DPM-Solver-1.
                - If steps % 3 == 2, we use (K - 1) steps of DPM-Solver-3 and 1 step of DPM-Solver-2.

        ============================================
        Args:
            order: A `int`. The max order for the solver (2 or 3).
            steps: A `int`. The total number of function evaluations (NFE).
            skip_type: A `str`. The type for the spacing of the time steps. We support three types:
                - 'logSNR': uniform logSNR for the time steps.
                - 'time_uniform': uniform time for the time steps. (**Recommended for high-resolutional data**.)
                - 'time_quadratic': quadratic time for the time steps. (Used in DDIM for low-resolutional data.)
            t_T: A `float`. The starting time of the sampling (default is T).
            t_0: A `float`. The ending time of the sampling (default is epsilon).
            device: A torch device.
        Returns:
            orders: A list of the solver order of each step.
        """
        if order == 3:
            K = steps // 3 + 1
            if steps % 3 == 0:
                orders = [3,] * (K - 2) + [2, 1]
            elif steps % 3 == 1:
                orders = [3,] * (K - 1) + [1]
            else:
                orders = [3,] * (K - 1) + [2]
        elif order == 2:
            if steps % 2 == 0:
                K = steps // 2
                orders = [2,] * K
            else:
                K = steps // 2 + 1
                orders = [2,] * (K - 1) + [1]
        elif order == 1:
            K = steps
            orders = [1,] * steps
        else:
            raise ValueError("'order' must be '1' or '2' or '3'.")
        if skip_type == 'logSNR':
            # To reproduce the results in DPM-Solver paper
            timesteps_outer = self.get_time_steps(skip_type, t_T, t_0, K, device)
        else:
            timesteps_outer = self.get_time_steps(skip_type, t_T, t_0, steps, device)[torch.cumsum(torch.tensor([0,] + orders), 0).to(device)]
        return timesteps_outer, orders

    def denoise_to_zero_fn(self, x, s):
        """
        Denoise at the final step, which is equivalent to solve the ODE from lambda_s to infty by first-order discretization. 
        """
        return self.data_prediction_fn(x, s)

    def dpm_solver_first_update(self, x, s, t, model_s=None, return_intermediate=False):
        """
        DPM-Solver-1 (equivalent to DDIM) from time `s` to time `t`.

        Args:
            x: A pytorch tensor. The initial value at time `s`.
            s: A pytorch tensor. The starting time, with the shape (1,).
            t: A pytorch tensor. The ending time, with the shape (1,).
            model_s: A pytorch tensor. The model function evaluated at time `s`.
                If `model_s` is None, we evaluate the model by `x` and `s`; otherwise we directly use it.
            return_intermediate: A `bool`. If true, also return the model value at time `s`.
        Returns:
            x_t: A pytorch tensor. The approximated solution at time `t`.
        """
        ns = self.noise_schedule
        dims = x.dim()
        lambda_s, lambda_t = ns.marginal_lambda(s), ns.marginal_lambda(t)
        h = lambda_t - lambda_s
        log_alpha_s, log_alpha_t = ns.marginal_log_mean_coeff(s), ns.marginal_log_mean_coeff(t)
        sigma_s, sigma_t = ns.marginal_std(s), ns.marginal_std(t)
        alpha_t = torch.exp(log_alpha_t)

        if self.algorithm_type == "dpmsolver++":
            phi_1 = torch.expm1(-h)
            if model_s is None:
                model_s = self.model_fn(x, s)
            x_t = (
                sigma_t / sigma_s * x
                - alpha_t * phi_1 * model_s
            )
            if return_intermediate:
                return x_t, {'model_s': model_s}
            else:
                return x_t
        else:
            phi_1 = torch.expm1(h)
            if model_s is None:
                model_s = self.model_fn(x, s)
            x_t = (
                torch.exp(log_alpha_t - log_alpha_s) * x
                - (sigma_t * phi_1) * model_s
            )
            if self.correcting_xt_fn is not None:
                x_0_pred = model_s
                x_t = self.correcting_xt_fn(x, x_0_pred, self.data,  x_t, self.noise_schedule.marginal_std(s))
            if return_intermediate:
                return x_t, {'model_s': model_s}
            else:
                return x_t

    def singlestep_dpm_solver_second_update(self, x, s, t, r1=0.5, model_s=None, return_intermediate=False, solver_type='dpmsolver'):
        """
        Singlestep solver DPM-Solver-2 from time `s` to time `t`.

        Args:
            x: A pytorch tensor. The initial value at time `s`.
            s: A pytorch tensor. The starting time, with the shape (1,).
            t: A pytorch tensor. The ending time, with the shape (1,).
            r1: A `float`. The hyperparameter of the second-order solver.
            model_s: A pytorch tensor. The model function evaluated at time `s`.
                If `model_s` is None, we evaluate the model by `x` and `s`; otherwise we directly use it.
            return_intermediate: A `bool`. If true, also return the model value at time `s` and `s1` (the intermediate time).
            solver_type: either 'dpmsolver' or 'taylor'. The type for the high-order solvers.
                The type slightly impacts the performance. We recommend to use 'dpmsolver' type.
        Returns:
            x_t: A pytorch tensor. The approximated solution at time `t`.
        """
        if solver_type not in ['dpmsolver', 'taylor']:
            raise ValueError("'solver_type' must be either 'dpmsolver' or 'taylor', got {}".format(solver_type))
        if r1 is None:
            r1 = 0.5
        ns = self.noise_schedule
        lambda_s, lambda_t = ns.marginal_lambda(s), ns.marginal_lambda(t)
        h = lambda_t - lambda_s
        lambda_s1 = lambda_s + r1 * h
        s1 = ns.inverse_lambda(lambda_s1)
        log_alpha_s, log_alpha_s1, log_alpha_t = ns.marginal_log_mean_coeff(s), ns.marginal_log_mean_coeff(s1), ns.marginal_log_mean_coeff(t)
        sigma_s, sigma_s1, sigma_t = ns.marginal_std(s), ns.marginal_std(s1), ns.marginal_std(t)
        alpha_s1, alpha_t = torch.exp(log_alpha_s1), torch.exp(log_alpha_t)

        if self.algorithm_type == "dpmsolver++":
            phi_11 = torch.expm1(-r1 * h)
            phi_1 = torch.expm1(-h)

            if model_s is None:
                model_s = self.model_fn(x, s)
            x_s1 = (
                (sigma_s1 / sigma_s) * x
                - (alpha_s1 * phi_11) * model_s
            )
            model_s1 = self.model_fn(x_s1, s1)
            if solver_type == 'dpmsolver':
                x_t = (
                    (sigma_t / sigma_s) * x
                    - (alpha_t * phi_1) * model_s
                    - (0.5 / r1) * (alpha_t * phi_1) * (model_s1 - model_s)
                )
            elif solver_type == 'taylor':
                x_t = (
                    (sigma_t / sigma_s) * x
                    - (alpha_t * phi_1) * model_s
                    + (1. / r1) * (alpha_t * (phi_1 / h + 1.)) * (model_s1 - model_s)
                )
        else:
            phi_11 = torch.expm1(r1 * h)
            phi_1 = torch.expm1(h)

            if model_s is None:
                model_s = self.model_fn(x, s)
            x_s1 = (
                torch.exp(log_alpha_s1 - log_alpha_s) * x
                - (sigma_s1 * phi_11) * model_s
            )
            model_s1 = self.model_fn(x_s1, s1)
            if solver_type == 'dpmsolver':
                x_t = (
                    torch.exp(log_alpha_t - log_alpha_s) * x
                    - (sigma_t * phi_1) * model_s
                    - (0.5 / r1) * (sigma_t * phi_1) * (model_s1 - model_s)
                )
            elif solver_type == 'taylor':
                x_t = (
                    torch.exp(log_alpha_t - log_alpha_s) * x
                    - (sigma_t * phi_1) * model_s
                    - (1. / r1) * (sigma_t * (phi_1 / h - 1.)) * (model_s1 - model_s)
                )
        if self.correcting_xt_fn is not None:
            x_0_pred = model_s
            x_t = self.correcting_xt_fn(x, x_0_pred, self.data, x_sample = x_t, std = self.noise_schedule.marginal_std(s))
        if return_intermediate:
            return x_t, {'model_s': model_s, 'model_s1': model_s1}
        else:
            return x_t

    def singlestep_dpm_solver_third_update(self, x, s, t, r1=1./3., r2=2./3., model_s=None, model_s1=None, return_intermediate=False, solver_type='dpmsolver'):
        """
        Singlestep solver DPM-Solver-3 from time `s` to time `t`.

        Args:
            x: A pytorch tensor. The initial value at time `s`.
            s: A pytorch tensor. The starting time, with the shape (1,).
            t: A pytorch tensor. The ending time, with the shape (1,).
            r1: A `float`. The hyperparameter of the third-order solver.
            r2: A `float`. The hyperparameter of the third-order solver.
            model_s: A pytorch tensor. The model function evaluated at time `s`.
                If `model_s` is None, we evaluate the model by `x` and `s`; otherwise we directly use it.
            model_s1: A pytorch tensor. The model function evaluated at time `s1` (the intermediate time given by `r1`).
                If `model_s1` is None, we evaluate the model at `s1`; otherwise we directly use it.
            return_intermediate: A `bool`. If true, also return the model value at time `s`, `s1` and `s2` (the intermediate times).
            solver_type: either 'dpmsolver' or 'taylor'. The type for the high-order solvers.
                The type slightly impacts the performance. We recommend to use 'dpmsolver' type.
        Returns:
            x_t: A pytorch tensor. The approximated solution at time `t`.
        """
        if solver_type not in ['dpmsolver', 'taylor']:
            raise ValueError("'solver_type' must be either 'dpmsolver' or 'taylor', got {}".format(solver_type))
        if r1 is None:
            r1 = 1. / 3.
        if r2 is None:
            r2 = 2. / 3.
        ns = self.noise_schedule
        lambda_s, lambda_t = ns.marginal_lambda(s), ns.marginal_lambda(t)
        h = lambda_t - lambda_s
        lambda_s1 = lambda_s + r1 * h
        lambda_s2 = lambda_s + r2 * h
        s1 = ns.inverse_lambda(lambda_s1)
        s2 = ns.inverse_lambda(lambda_s2)
        log_alpha_s, log_alpha_s1, log_alpha_s2, log_alpha_t = ns.marginal_log_mean_coeff(s), ns.marginal_log_mean_coeff(s1), ns.marginal_log_mean_coeff(s2), ns.marginal_log_mean_coeff(t)
        sigma_s, sigma_s1, sigma_s2, sigma_t = ns.marginal_std(s), ns.marginal_std(s1), ns.marginal_std(s2), ns.marginal_std(t)
        alpha_s1, alpha_s2, alpha_t = torch.exp(log_alpha_s1), torch.exp(log_alpha_s2), torch.exp(log_alpha_t)

        if self.algorithm_type == "dpmsolver++":
            phi_11 = torch.expm1(-r1 * h)
            phi_12 = torch.expm1(-r2 * h)
            phi_1 = torch.expm1(-h)
            phi_22 = torch.expm1(-r2 * h) / (r2 * h) + 1.
            phi_2 = phi_1 / h + 1.
            phi_3 = phi_2 / h - 0.5

            if model_s is None:
                model_s = self.model_fn(x, s)
            if model_s1 is None:
                x_s1 = (
                    (sigma_s1 / sigma_s) * x
                    - (alpha_s1 * phi_11) * model_s
                )
                model_s1 = self.model_fn(x_s1, s1)
            x_s2 = (
                (sigma_s2 / sigma_s) * x
                - (alpha_s2 * phi_12) * model_s
                + r2 / r1 * (alpha_s2 * phi_22) * (model_s1 - model_s)
            )
            model_s2 = self.model_fn(x_s2, s2)
            if solver_type == 'dpmsolver':
                x_t = (
                    (sigma_t / sigma_s) * x
                    - (alpha_t * phi_1) * model_s
                    + (1. / r2) * (alpha_t * phi_2) * (model_s2 - model_s)
                )
            elif solver_type == 'taylor':
                D1_0 = (1. / r1) * (model_s1 - model_s)
                D1_1 = (1. / r2) * (model_s2 - model_s)
                D1 = (r2 * D1_0 - r1 * D1_1) / (r2 - r1)
                D2 = 2. * (D1_1 - D1_0) / (r2 - r1)
                x_t = (
                    (sigma_t / sigma_s) * x
                    - (alpha_t * phi_1) * model_s
                    + (alpha_t * phi_2) * D1
                    - (alpha_t * phi_3) * D2
                )
        else:
            phi_11 = torch.expm1(r1 * h)
            phi_12 = torch.expm1(r2 * h)
            phi_1 = torch.expm1(h)
            phi_22 = torch.expm1(r2 * h) / (r2 * h) - 1.
            phi_2 = phi_1 / h - 1.
            phi_3 = phi_2 / h - 0.5

            if model_s is None:
                model_s = self.model_fn(x, s)
            if model_s1 is None:
                x_s1 = (
                    (torch.exp(log_alpha_s1 - log_alpha_s)) * x
                    - (sigma_s1 * phi_11) * model_s
                )
                model_s1 = self.model_fn(x_s1, s1)
            x_s2 = (
                (torch.exp(log_alpha_s2 - log_alpha_s)) * x
                - (sigma_s2 * phi_12) * model_s
                - r2 / r1 * (sigma_s2 * phi_22) * (model_s1 - model_s)
            )
            model_s2 = self.model_fn(x_s2, s2)
            if solver_type == 'dpmsolver':
                x_t = (
                    (torch.exp(log_alpha_t - log_alpha_s)) * x
                    - (sigma_t * phi_1) * model_s
                    - (1. / r2) * (sigma_t * phi_2) * (model_s2 - model_s)
                )
            elif solver_type == 'taylor':
                D1_0 = (1. / r1) * (model_s1 - model_s)
                D1_1 = (1. / r2) * (model_s2 - model_s)
                D1 = (r2 * D1_0 - r1 * D1_1) / (r2 - r1)
                D2 = 2. * (D1_1 - D1_0) / (r2 - r1)
                x_t = (
                    (torch.exp(log_alpha_t - log_alpha_s)) * x
                    - (sigma_t * phi_1) * model_s
                    - (sigma_t * phi_2) * D1
                    - (sigma_t * phi_3) * D2
                )
        if self.correcting_xt_fn is not None:
            x_0_pred = model_s
            x_t = self.correcting_xt_fn(x, x_0_pred, self.data, x_sample = x_t, std = self.noise_schedule.marginal_std(s))
        if return_intermediate:
            return x_t, {'model_s': model_s, 'model_s1': model_s1, 'model_s2': model_s2}
        else:
            return x_t

    def multistep_dpm_solver_second_update(self, x, model_prev_list, t_prev_list, t, solver_type="dpmsolver"):
        """
        Multistep solver DPM-Solver-2 from time `t_prev_list[-1]` to time `t`.

        Args:
            x: A pytorch tensor. The initial value at time `s`.
            model_prev_list: A list of pytorch tensor. The previous computed model values.
            t_prev_list: A list of pytorch tensor. The previous times, each time has the shape (1,)
            t: A pytorch tensor. The ending time, with the shape (1,).
            solver_type: either 'dpmsolver' or 'taylor'. The type for the high-order solvers.
                The type slightly impacts the performance. We recommend to use 'dpmsolver' type.
        Returns:
            x_t: A pytorch tensor. The approximated solution at time `t`.
        """
        if solver_type not in ['dpmsolver', 'taylor']:
            raise ValueError("'solver_type' must be either 'dpmsolver' or 'taylor', got {}".format(solver_type))
        ns = self.noise_schedule
        model_prev_1, model_prev_0 = model_prev_list[-2], model_prev_list[-1]
        t_prev_1, t_prev_0 = t_prev_list[-2], t_prev_list[-1]
        lambda_prev_1, lambda_prev_0, lambda_t = ns.marginal_lambda(t_prev_1), ns.marginal_lambda(t_prev_0), ns.marginal_lambda(t)
        log_alpha_prev_0, log_alpha_t = ns.marginal_log_mean_coeff(t_prev_0), ns.marginal_log_mean_coeff(t)
        sigma_prev_0, sigma_t = ns.marginal_std(t_prev_0), ns.marginal_std(t)
        alpha_t = torch.exp(log_alpha_t)
        h_0 = lambda_prev_0 - lambda_prev_1
        h = lambda_t - lambda_prev_0
        r0 = h_0 / h
        D1_0 = (1. / r0) * (model_prev_0 - model_prev_1)
        if self.algorithm_type == "dpmsolver++":
            phi_1 = torch.expm1(-h)
            if solver_type == 'dpmsolver':
                x_t = (
                    (sigma_t / sigma_prev_0) * x
                    - (alpha_t * phi_1) * model_prev_0
                    - 0.5 * (alpha_t * phi_1) * D1_0
                )
            elif solver_type == 'taylor':
                x_t = (
                    (sigma_t / sigma_prev_0) * x
                    - (alpha_t * phi_1) * model_prev_0
                    + (alpha_t * (phi_1 / h + 1.)) * D1_0
                )
        else:
            phi_1 = torch.expm1(h)
            if solver_type == 'dpmsolver':
                x_t = (
                    (torch.exp(log_alpha_t - log_alpha_prev_0)) * x
                    - (sigma_t * phi_1) * model_prev_0
                    - 0.5 * (sigma_t * phi_1) * D1_0
                )
            elif solver_type == 'taylor':
                x_t = (
                    (torch.exp(log_alpha_t - log_alpha_prev_0)) * x
                    - (sigma_t * phi_1) * model_prev_0
                    - (sigma_t * (phi_1 / h - 1.)) * D1_0
                )
        if self.correcting_xt_fn is not None:

            x_0_pred = model_prev_0
            x_t = self.correcting_xt_fn(x, x_0_pred, self.data, x_sample = x_t, std = self.noise_schedule.marginal_std(t_prev_0))
        return x_t

    def multistep_dpm_solver_third_update(self, x, model_prev_list, t_prev_list, t, solver_type='dpmsolver'):
        """
        Multistep solver DPM-Solver-3 from time `t_prev_list[-1]` to time `t`.

        Args:
            x: A pytorch tensor. The initial value at time `s`.
            model_prev_list: A list of pytorch tensor. The previous computed model values.
            t_prev_list: A list of pytorch tensor. The previous times, each time has the shape (1,)
            t: A pytorch tensor. The ending time, with the shape (1,).
            solver_type: either 'dpmsolver' or 'taylor'. The type for the high-order solvers.
                The type slightly impacts the performance. We recommend to use 'dpmsolver' type.
        Returns:
            x_t: A pytorch tensor. The approximated solution at time `t`.
        """
        ns = self.noise_schedule
        model_prev_2, model_prev_1, model_prev_0 = model_prev_list
        t_prev_2, t_prev_1, t_prev_0 = t_prev_list
        lambda_prev_2, lambda_prev_1, lambda_prev_0, lambda_t = ns.marginal_lambda(t_prev_2), ns.marginal_lambda(t_prev_1), ns.marginal_lambda(t_prev_0), ns.marginal_lambda(t)
        log_alpha_prev_0, log_alpha_t = ns.marginal_log_mean_coeff(t_prev_0), ns.marginal_log_mean_coeff(t)
        sigma_prev_0, sigma_t = ns.marginal_std(t_prev_0), ns.marginal_std(t)
        alpha_t = torch.exp(log_alpha_t)

        h_1 = lambda_prev_1 - lambda_prev_2
        h_0 = lambda_prev_0 - lambda_prev_1
        h = lambda_t - lambda_prev_0
        r0, r1 = h_0 / h, h_1 / h
        D1_0 = (1. / r0) * (model_prev_0 - model_prev_1)
        D1_1 = (1. / r1) * (model_prev_1 - model_prev_2)
        D1 = D1_0 + (r0 / (r0 + r1)) * (D1_0 - D1_1)
        D2 = (1. / (r0 + r1)) * (D1_0 - D1_1)
        if self.algorithm_type == "dpmsolver++":
            phi_1 = torch.expm1(-h)
            phi_2 = phi_1 / h + 1.
            phi_3 = phi_2 / h - 0.5
            x_t = (
                (sigma_t / sigma_prev_0) * x
                - (alpha_t * phi_1) * model_prev_0
                + (alpha_t * phi_2) * D1
                - (alpha_t * phi_3) * D2
            )
        else:
            phi_1 = torch.expm1(h)
            phi_2 = phi_1 / h - 1.
            phi_3 = phi_2 / h - 0.5
            x_t = (
                (torch.exp(log_alpha_t - log_alpha_prev_0)) * x
                - (sigma_t * phi_1) * model_prev_0
                - (sigma_t * phi_2) * D1
                - (sigma_t * phi_3) * D2
            )
        if self.correcting_xt_fn is not None:
            x_0_pred = model_prev_0
            x_t = self.correcting_xt_fn(x, x_0_pred, self.data, x_sample = x_t, std = self.noise_schedule.marginal_std(t_prev_0))
        return x_t

    def singlestep_dpm_solver_update(self, x, s, t, order, model_s=None, return_intermediate=False, solver_type='dpmsolver', r1=None, r2=None):
        """
        Singlestep DPM-Solver with the order `order` from time `s` to time `t`.

        Args:
            x: A pytorch tensor. The initial value at time `s`.
            s: A pytorch tensor. The starting time, with the shape (1,).
            t: A pytorch tensor. The ending time, with the shape (1,).
            order: A `int`. The order of DPM-Solver. We only support order == 1 or 2 or 3.
            return_intermediate: A `bool`. If true, also return the model value at time `s`, `s1` and `s2` (the intermediate times).
            solver_type: either 'dpmsolver' or 'taylor'. The type for the high-order solvers.
                The type slightly impacts the performance. We recommend to use 'dpmsolver' type.
            r1: A `float`. The hyperparameter of the second-order or third-order solver.
            r2: A `float`. The hyperparameter of the third-order solver.
        Returns:
            x_t: A pytorch tensor. The approximated solution at time `t`.
        """
        if order == 1:
            return self.dpm_solver_first_update(x, s, t, model_s, return_intermediate=return_intermediate)
        elif order == 2:
            return self.singlestep_dpm_solver_second_update(x, s, t, model_s, return_intermediate=return_intermediate, solver_type=solver_type, r1=r1)
        elif order == 3:
            return self.singlestep_dpm_solver_third_update(x, s, t, model_s, return_intermediate=return_intermediate, solver_type=solver_type, r1=r1, r2=r2)
        else:
            raise ValueError("Solver order must be 1 or 2 or 3, got {}".format(order))

    def multistep_dpm_solver_update(self, x, model_prev_list, t_prev_list, t, order, solver_type='dpmsolver'):
        """
        Multistep DPM-Solver with the order `order` from time `t_prev_list[-1]` to time `t`.

        Args:
            x: A pytorch tensor. The initial value at time `s`.
            model_prev_list: A list of pytorch tensor. The previous computed model values.
            t_prev_list: A list of pytorch tensor. The previous times, each time has the shape (1,)
            t: A pytorch tensor. The ending time, with the shape (1,).
            order: A `int`. The order of DPM-Solver. We only support order == 1 or 2 or 3.
            solver_type: either 'dpmsolver' or 'taylor'. The type for the high-order solvers.
                The type slightly impacts the performance. We recommend to use 'dpmsolver' type.
        Returns:
            x_t: A pytorch tensor. The approximated solution at time `t`.
        """
        if order == 1:
            return self.dpm_solver_first_update(x, t_prev_list[-1], t, model_s=model_prev_list[-1])
        elif order == 2:
            return self.multistep_dpm_solver_second_update(x, model_prev_list, t_prev_list, t, solver_type=solver_type)
        elif order == 3:
            return self.multistep_dpm_solver_third_update(x, model_prev_list, t_prev_list, t, solver_type=solver_type)
        else:
            raise ValueError("Solver order must be 1 or 2 or 3, got {}".format(order))

    def dpm_solver_adaptive(self, x, order, t_T, t_0, h_init=0.05, atol=0.0078, rtol=0.05, theta=0.9, t_err=1e-5, solver_type='dpmsolver'):
        """
        The adaptive step size solver based on singlestep DPM-Solver.

        Args:
            x: A pytorch tensor. The initial value at time `t_T`.
            order: A `int`. The (higher) order of the solver. We only support order == 2 or 3.
            t_T: A `float`. The starting time of the sampling (default is T).
            t_0: A `float`. The ending time of the sampling (default is epsilon).
            h_init: A `float`. The initial step size (for logSNR).
            atol: A `float`. The absolute tolerance of the solver. For image data, the default setting is 0.0078, followed [1].
            rtol: A `float`. The relative tolerance of the solver. The default setting is 0.05.
            theta: A `float`. The safety hyperparameter for adapting the step size. The default setting is 0.9, followed [1].
            t_err: A `float`. The tolerance for the time. We solve the diffusion ODE until the absolute error between the 
                current time and `t_0` is less than `t_err`. The default setting is 1e-5.
            solver_type: either 'dpmsolver' or 'taylor'. The type for the high-order solvers.
                The type slightly impacts the performance. We recommend to use 'dpmsolver' type.
        Returns:
            x_0: A pytorch tensor. The approximated solution at time `t_0`.

        [1] A. Jolicoeur-Martineau, K. Li, R. Piché-Taillefer, T. Kachman, and I. Mitliagkas, "Gotta go fast when generating data with score-based models," arXiv preprint arXiv:2105.14080, 2021.
        """
        ns = self.noise_schedule
        s = t_T * torch.ones((1,)).to(x)
        lambda_s = ns.marginal_lambda(s)
        lambda_0 = ns.marginal_lambda(t_0 * torch.ones_like(s).to(x))
        h = h_init * torch.ones_like(s).to(x)
        x_prev = x
        nfe = 0
        if order == 2:
            r1 = 0.5
            lower_update = lambda x, s, t: self.dpm_solver_first_update(x, s, t, return_intermediate=True)
            higher_update = lambda x, s, t, **kwargs: self.singlestep_dpm_solver_second_update(x, s, t, r1=r1, solver_type=solver_type, **kwargs)
        elif order == 3:
            r1, r2 = 1. / 3., 2. / 3.
            lower_update = lambda x, s, t: self.singlestep_dpm_solver_second_update(x, s, t, r1=r1, return_intermediate=True, solver_type=solver_type)
            higher_update = lambda x, s, t, **kwargs: self.singlestep_dpm_solver_third_update(x, s, t, r1=r1, r2=r2, solver_type=solver_type, **kwargs)
        else:
            raise ValueError("For adaptive step size solver, order must be 2 or 3, got {}".format(order))
        while torch.abs((s - t_0)).mean() > t_err:
            t = ns.inverse_lambda(lambda_s + h)
            x_lower, lower_noise_kwargs = lower_update(x, s, t)
            x_higher = higher_update(x, s, t, **lower_noise_kwargs)
            delta = torch.max(torch.ones_like(x).to(x) * atol, rtol * torch.max(torch.abs(x_lower), torch.abs(x_prev)))
            norm_fn = lambda v: torch.sqrt(torch.square(v.reshape((v.shape[0], -1))).mean(dim=-1, keepdim=True))
            E = norm_fn((x_higher - x_lower) / delta).max()
            if torch.all(E <= 1.):
                x = x_higher
                s = t
                x_prev = x_lower
                lambda_s = ns.marginal_lambda(s)
            h = torch.min(theta * h * torch.float_power(E, -1. / order).float(), lambda_0 - lambda_s)
            nfe += order
        print('adaptive solver nfe', nfe)
        return x

    def add_noise(self, x, t, noise=None):
        """
        Compute the noised input xt = alpha_t * x + sigma_t * noise. 

        Args:
            x: A `torch.Tensor` with shape `(batch_size, *shape)`.
            t: A `torch.Tensor` with shape `(t_size,)`.
        Returns:
            xt with shape `(t_size, batch_size, *shape)`.
        """
        alpha_t, sigma_t = self.noise_schedule.marginal_alpha(t), self.noise_schedule.marginal_std(t)
        if noise is None:
            noise = torch.randn((t.shape[0], *x.shape), device=x.device)
        x = x.reshape((-1, *x.shape))
        xt = expand_dims(alpha_t, x.dim()) * x + expand_dims(sigma_t, x.dim()) * noise
        if t.shape[0] == 1:
            return xt.squeeze(0)
        else:
            return xt

    def inverse(self, x, steps=20, t_start=None, t_end=None, order=2, skip_type='time_uniform',
        method='multistep', lower_order_final=True, denoise_to_zero=False, solver_type='dpmsolver',
        atol=0.0078, rtol=0.05, return_intermediate=False,
    ):
        """
        Inverse the sample `x` from time `t_start` to `t_end` by DPM-Solver.
        For discrete-time DPMs, we use `t_start=1/N`, where `N` is the total time steps during training.
        """
        t_0 = 1. / self.noise_schedule.total_N if t_start is None else t_start
        t_T = self.noise_schedule.T if t_end is None else t_end
        assert t_0 > 0 and t_T > 0, "Time range needs to be greater than 0. For discrete-time DPMs, it needs to be in [1 / N, 1], where N is the length of betas array"
        return self.sample(x, steps=steps, t_start=t_0, t_end=t_T, order=order, skip_type=skip_type,
            method=method, lower_order_final=lower_order_final, denoise_to_zero=denoise_to_zero, solver_type=solver_type,
            atol=atol, rtol=rtol, return_intermediate=return_intermediate)

    def sample(self, x, steps=20, t_start=None, t_end=None, order=2, skip_type='time_uniform',
        method='multistep', lower_order_final=True, denoise_to_zero=False, solver_type='dpmsolver',
        atol=0.0078, rtol=0.05, return_intermediate=False,
    ):
        """
        Compute the sample at time `t_end` by DPM-Solver, given the initial `x` at time `t_start`.

        =====================================================

        We support the following algorithms for both noise prediction model and data prediction model:
            - 'singlestep':
                Singlestep DPM-Solver (i.e. "DPM-Solver-fast" in the paper), which combines different orders of singlestep DPM-Solver. 
                We combine all the singlestep solvers with order <= `order` to use up all the function evaluations (steps).
                The total number of function evaluations (NFE) == `steps`.
                Given a fixed NFE == `steps`, the sampling procedure is:
                    - If `order` == 1:
                        - Denote K = steps. We use K steps of DPM-Solver-1 (i.e. DDIM).
                    - If `order` == 2:
                        - Denote K = (steps // 2) + (steps % 2). We take K intermediate time steps for sampling.
                        - If steps % 2 == 0, we use K steps of singlestep DPM-Solver-2.
                        - If steps % 2 == 1, we use (K - 1) steps of singlestep DPM-Solver-2 and 1 step of DPM-Solver-1.
                    - If `order` == 3:
                        - Denote K = (steps // 3 + 1). We take K intermediate time steps for sampling.
                        - If steps % 3 == 0, we use (K - 2) steps of singlestep DPM-Solver-3, and 1 step of singlestep DPM-Solver-2 and 1 step of DPM-Solver-1.
                        - If steps % 3 == 1, we use (K - 1) steps of singlestep DPM-Solver-3 and 1 step of DPM-Solver-1.
                        - If steps % 3 == 2, we use (K - 1) steps of singlestep DPM-Solver-3 and 1 step of singlestep DPM-Solver-2.
            - 'multistep':
                Multistep DPM-Solver with the order of `order`. The total number of function evaluations (NFE) == `steps`.
                We initialize the first `order` values by lower order multistep solvers.
                Given a fixed NFE == `steps`, the sampling procedure is:
                    Denote K = steps.
                    - If `order` == 1:
                        - We use K steps of DPM-Solver-1 (i.e. DDIM).
                    - If `order` == 2:
                        - We firstly use 1 step of DPM-Solver-1, then use (K - 1) step of multistep DPM-Solver-2.
                    - If `order` == 3:
                        - We firstly use 1 step of DPM-Solver-1, then 1 step of multistep DPM-Solver-2, then (K - 2) step of multistep DPM-Solver-3.
            - 'singlestep_fixed':
                Fixed order singlestep DPM-Solver (i.e. DPM-Solver-1 or singlestep DPM-Solver-2 or singlestep DPM-Solver-3).
                We use singlestep DPM-Solver-`order` for `order`=1 or 2 or 3, with total [`steps` // `order`] * `order` NFE.
            - 'adaptive':
                Adaptive step size DPM-Solver (i.e. "DPM-Solver-12" and "DPM-Solver-23" in the paper).
                We ignore `steps` and use adaptive step size DPM-Solver with a higher order of `order`.
                You can adjust the absolute tolerance `atol` and the relative tolerance `rtol` to balance the computatation costs
                (NFE) and the sample quality.
                    - If `order` == 2, we use DPM-Solver-12 which combines DPM-Solver-1 and singlestep DPM-Solver-2.
                    - If `order` == 3, we use DPM-Solver-23 which combines singlestep DPM-Solver-2 and singlestep DPM-Solver-3.

        =====================================================

        Some advices for choosing the algorithm:
            - For **unconditional sampling** or **guided sampling with small guidance scale** by DPMs:
                Use singlestep DPM-Solver or DPM-Solver++ ("DPM-Solver-fast" in the paper) with `order = 3`.
                e.g., DPM-Solver:
                    >>> dpm_solver = DPM_Solver(model_fn, noise_schedule, algorithm_type="dpmsolver")
                    >>> x_sample = dpm_solver.sample(x, steps=steps, t_start=t_start, t_end=t_end, order=3,
                            skip_type='time_uniform', method='singlestep')
                e.g., DPM-Solver++:
                    >>> dpm_solver = DPM_Solver(model_fn, noise_schedule, algorithm_type="dpmsolver++")
                    >>> x_sample = dpm_solver.sample(x, steps=steps, t_start=t_start, t_end=t_end, order=3,
                            skip_type='time_uniform', method='singlestep')
            - For **guided sampling with large guidance scale** by DPMs:
                Use multistep DPM-Solver with `algorithm_type="dpmsolver++"` and `order = 2`.
                e.g.
                    >>> dpm_solver = DPM_Solver(model_fn, noise_schedule, algorithm_type="dpmsolver++")
                    >>> x_sample = dpm_solver.sample(x, steps=steps, t_start=t_start, t_end=t_end, order=2,
                            skip_type='time_uniform', method='multistep')

        We support three types of `skip_type`:
            - 'logSNR': uniform logSNR for the time steps. **Recommended for low-resolutional images**
            - 'time_uniform': uniform time for the time steps. **Recommended for high-resolutional images**.
            - 'time_quadratic': quadratic time for the time steps.

        =====================================================
        Args:
            x: A pytorch tensor. The initial value at time `t_start`
                e.g. if `t_start` == T, then `x` is a sample from the standard normal distribution.
            steps: A `int`. The total number of function evaluations (NFE).
            t_start: A `float`. The starting time of the sampling.
                If `T` is None, we use self.noise_schedule.T (default is 1.0).
            t_end: A `float`. The ending time of the sampling.
                If `t_end` is None, we use 1. / self.noise_schedule.total_N.
                e.g. if total_N == 1000, we have `t_end` == 1e-3.
                For discrete-time DPMs:
                    - We recommend `t_end` == 1. / self.noise_schedule.total_N.
                For continuous-time DPMs:
                    - We recommend `t_end` == 1e-3 when `steps` <= 15; and `t_end` == 1e-4 when `steps` > 15.
            order: A `int`. The order of DPM-Solver.
            skip_type: A `str`. The type for the spacing of the time steps. 'time_uniform' or 'logSNR' or 'time_quadratic'.
            method: A `str`. The method for sampling. 'singlestep' or 'multistep' or 'singlestep_fixed' or 'adaptive'.
            denoise_to_zero: A `bool`. Whether to denoise to time 0 at the final step.
                Default is `False`. If `denoise_to_zero` is `True`, the total NFE is (`steps` + 1).

                This trick is firstly proposed by DDPM (https://arxiv.org/abs/2006.11239) and
                score_sde (https://arxiv.org/abs/2011.13456). Such trick can improve the FID
                for diffusion models sampling by diffusion SDEs for low-resolutional images
                (such as CIFAR-10). However, we observed that such trick does not matter for
                high-resolutional images. As it needs an additional NFE, we do not recommend
                it for high-resolutional images.
            lower_order_final: A `bool`. Whether to use lower order solvers at the final steps.
                Only valid for `method=multistep` and `steps < 15`. We empirically find that
                this trick is a key to stabilizing the sampling by DPM-Solver with very few steps
                (especially for steps <= 10). So we recommend to set it to be `True`.
            solver_type: A `str`. The taylor expansion type for the solver. `dpmsolver` or `taylor`. We recommend `dpmsolver`.
            atol: A `float`. The absolute tolerance of the adaptive step size solver. Valid when `method` == 'adaptive'.
            rtol: A `float`. The relative tolerance of the adaptive step size solver. Valid when `method` == 'adaptive'.
            return_intermediate: A `bool`. Whether to save the xt at each step.
                When set to `True`, method returns a tuple (x0, intermediates); when set to False, method returns only x0.
        Returns:
            x_end: A pytorch tensor. The approximated solution at time `t_end`.

        """
        t_0 = 1. / self.noise_schedule.total_N if t_end is None else t_end
        t_T = self.noise_schedule.T if t_start is None else t_start
        assert t_0 > 0 and t_T > 0, "Time range needs to be greater than 0. For discrete-time DPMs, it needs to be in [1 / N, 1], where N is the length of betas array"
        if return_intermediate:
            assert method in ['multistep', 'singlestep', 'singlestep_fixed'], "Cannot use adaptive solver when saving intermediate values"
        if self.correcting_xt_fn is not None:
            assert method in ['multistep', 'singlestep', 'singlestep_fixed'], "Cannot use adaptive solver when correcting_xt_fn is not None"
        device = x.device
        intermediates = []
        with torch.no_grad():
            if method == 'adaptive':
                with torch.enable_grad():
                    x = self.dpm_solver_adaptive(x, order=order, t_T=t_T, t_0=t_0, atol=atol, rtol=rtol, solver_type=solver_type)          
            elif method == 'multistep':
                assert steps >= order
                timesteps = self.get_time_steps(skip_type=skip_type, t_T=t_T, t_0=t_0, N=steps, device=device)
                assert timesteps.shape[0] - 1 == steps
                # Init the initial values.
                step = 0
                t = timesteps[step]
                t_prev_list = [t]
                with torch.enable_grad():
                    model_prev_list = [self.model_fn(x, t)]
                    if return_intermediate:
                        intermediates.append(x)
                # Init the first `order` values by lower order multistep DPM-Solver.
                for step in range(1, order):
                    t = timesteps[step]
                    x = self.multistep_dpm_solver_update(x, model_prev_list, t_prev_list, t, step, solver_type=solver_type).requires_grad_(True)###order =1
                    if return_intermediate:
                        intermediates.append(x)
                    t_prev_list.append(t)
                with torch.enable_grad():
                    model_prev_list.append(self.model_fn(x, t))
                # Compute the remaining values by `order`-th order multistep DPM-Solver.
                for step in range(order, steps + 1):
                    t = timesteps[step]
                    # We only use lower order for steps < 10
                    if lower_order_final and steps < 10:
                        step_order = min(order, steps + 1 - step)
                    else:
                        step_order = order   
                    x = self.multistep_dpm_solver_update(x, model_prev_list, t_prev_list, t, step_order, solver_type=solver_type).requires_grad_(True)
                    if return_intermediate:
                        intermediates.append(x)
                    for i in range(order - 1):
                        t_prev_list[i] = t_prev_list[i + 1]
                        with torch.enable_grad():
                            model_prev_list[i] = model_prev_list[i + 1]
                    t_prev_list[-1] = t
                    # We do not need to evaluate the final model value.
                    if step < steps:
                        with torch.enable_grad():
                            model_prev_list[-1] = self.model_fn(x, t)
            elif method in ['singlestep', 'singlestep_fixed']:
                if method == 'singlestep':
                    timesteps_outer, orders = self.get_orders_and_timesteps_for_singlestep_solver(steps=steps, order=order, skip_type=skip_type, t_T=t_T, t_0=t_0, device=device)
                elif method == 'singlestep_fixed':
                    K = steps // order
                    orders = [order,] * K
                    timesteps_outer = self.get_time_steps(skip_type=skip_type, t_T=t_T, t_0=t_0, N=K, device=device)
                for step, order in enumerate(orders):
                    s, t = timesteps_outer[step], timesteps_outer[step + 1]
                    timesteps_inner = self.get_time_steps(skip_type=skip_type, t_T=s.item(), t_0=t.item(), N=order, device=device)
                    lambda_inner = self.noise_schedule.marginal_lambda(timesteps_inner)
                    h = lambda_inner[-1] - lambda_inner[0]
                    r1 = None if order <= 1 else (lambda_inner[1] - lambda_inner[0]) / h
                    r2 = None if order <= 2 else (lambda_inner[2] - lambda_inner[0]) / h
                    with torch.enable_grad():
                        x = self.singlestep_dpm_solver_update(x, s, t, order, solver_type=solver_type, r1=r1, r2=r2)
                    if return_intermediate:
                        intermediates.append(x)
            else:
                raise ValueError("Got wrong method {}".format(method))
            if denoise_to_zero:
                t = torch.ones((1,)).to(device) * t_0
                with torch.enable_grad():
                    x = self.denoise_to_zero_fn(x, t)
                if return_intermediate:
                    intermediates.append(x)
            if return_intermediate:
                return x, intermediates
            else:
                return x



#############################################################
# other utility functions
#############################################################

def interpolate_fn(x, xp, yp):
    """
    A piecewise linear function y = f(x), using xp and yp as keypoints.
    We implement f(x) in a differentiable way (i.e. applicable for autograd).
    The function f(x) is well-defined for all x-axis. (For x beyond the bounds of xp, we use the outmost points of xp to define the linear function.)

    Args:
        x: PyTorch tensor with shape [N, C], where N is the batch size, C is the number of channels (we use C = 1 for DPM-Solver).
        xp: PyTorch tensor with shape [C, K], where K is the number of keypoints.
        yp: PyTorch tensor with shape [C, K].
    Returns:
        The function values f(x), with shape [N, C].
    """
    N, K = x.shape[0], xp.shape[1]
    all_x = torch.cat([x.unsqueeze(2), xp.unsqueeze(0).repeat((N, 1, 1))], dim=2)
    sorted_all_x, x_indices = torch.sort(all_x, dim=2)
    x_idx = torch.argmin(x_indices, dim=2)
    cand_start_idx = x_idx - 1
    start_idx = torch.where(
        torch.eq(x_idx, 0),
        torch.tensor(1, device=x.device),
        torch.where(
            torch.eq(x_idx, K), torch.tensor(K - 2, device=x.device), cand_start_idx,
        ),
    )
    end_idx = torch.where(torch.eq(start_idx, cand_start_idx), start_idx + 2, start_idx + 1)
    start_x = torch.gather(sorted_all_x, dim=2, index=start_idx.unsqueeze(2)).squeeze(2)
    end_x = torch.gather(sorted_all_x, dim=2, index=end_idx.unsqueeze(2)).squeeze(2)
    start_idx2 = torch.where(
        torch.eq(x_idx, 0),
        torch.tensor(0, device=x.device),
        torch.where(
            torch.eq(x_idx, K), torch.tensor(K - 2, device=x.device), cand_start_idx,
        ),
    )
    y_positions_expanded = yp.unsqueeze(0).expand(N, -1, -1)
    start_y = torch.gather(y_positions_expanded, dim=2, index=start_idx2.unsqueeze(2)).squeeze(2)
    end_y = torch.gather(y_positions_expanded, dim=2, index=(start_idx2 + 1).unsqueeze(2)).squeeze(2)
    cand = start_y + (x - start_x) * (end_y - start_y) / (end_x - start_x)
    return cand


def expand_dims(v, dims):
    """
    Expand the tensor `v` to the dim `dims`.

    Args:
        `v`: a PyTorch tensor with shape [N].
        `dim`: a `int`.
    Returns:
        a PyTorch tensor with shape [N, 1, 1, ..., 1] and the total dimension is `dims`.
    """
    return v[(...,) + (None,)*(dims - 1)]