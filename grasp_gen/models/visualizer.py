import os
import json
from loguru import logger
import torch
import torch.nn as nn
import numpy as np
from einops import rearrange
import trimesh
import pickle
from omegaconf import DictConfig
from plotly import graph_objects as go
from typing import Any
import random
from utils.misc import random_str
from utils.registry import Registry
from utils.visualize import frame2gif, render_prox_scene, render_scannet_path
from utils.visualize import create_trimesh_nodes_path, create_trimesh_node
from utils.handmodel import get_handmodel
from utils.plotly_utils import plot_mesh
from utils.rot6d import rot_to_orthod6d, robust_compute_rotation_matrix_from_ortho6d, random_rot
from tqdm import tqdm
import pdb, copy
import contextlib
# from models.latent_discriminator import DiscriminatorTrainer
# from ours_utils.hand_transformation import apply_transformations, normalize_trans_torch, normalize_param_torch, normalize_rot_torch, trans_normalize, angle_normalize

VISUALIZER = Registry('Visualizer')

def save_ckpt(model: torch.nn.Module, epoch: int, step: int, path: str, save_scene_model: bool) -> None:
    """ Save current model and corresponding data

    Args:
        model: best model
        epoch: best epoch
        step: current step
        path: save path
        save_scene_model: if save scene_model
    """
    saved_state_dict = {}
    model_state_dict = model.state_dict()
    for key in model_state_dict:
        ## if use frozen pretrained scene model, we can avoid saving scene model to save space
        if 'scene_model' in key and not save_scene_model:
            continue

        saved_state_dict[key] = model_state_dict[key]
    print('model saving!!!!!!')
    torch.save({
        'model': saved_state_dict,
        'epoch': epoch, 'step': step,
    }, path)

@VISUALIZER.register()
@torch.no_grad()
class PoseGenVisualizer():
    def __init__(self, cfg: DictConfig) -> None:
        """ Visual evaluation class for pose generation task.

        Args:
            cfg: visuzalizer configuration
        """
        self.vis_case_num = cfg.vis_case_num
        self.ksample = cfg.ksample
        self.vis_denoising = cfg.vis_denoising
        self.save_mesh = cfg.save_mesh

    def visualize(
        self, 
        model: torch.nn.Module, 
        dataloader: torch.utils.data.DataLoader, 
        save_dir: str,
    ) -> None:
        """ Visualize method

        Args:
            model: diffusion model
            dataloader: test dataloader
            save_dir: save directory of rendering images
        """
        model.eval()
        device = model.device
        
        cnt = 0
        for data in dataloader:
            for key in data:
                if torch.is_tensor(data[key]):
                    data[key] = data[key].to(device)
            data['normalizer'] = dataloader.dataset.normalizer
            
            ksample = 1 if self.vis_denoising else self.ksample
            outputs = model.sample(data, k=ksample) # <B, k, T, D>
            
            for i in range(outputs.shape[0]):
                scene_id = data['scene_id'][i]
                cam_tran = data['cam_tran'][i]
                gender = data['gender'][i]
                
                origin_cam_tran = data['origin_cam_tran'][i]
                scene_trans = cam_tran @ np.linalg.inv(origin_cam_tran) # scene_T @ origin_cam_T = cur_cam_T
                scene_mesh = dataloader.dataset.scene_meshes[scene_id].copy()
                scene_mesh.apply_transform(scene_trans)

                ## calculate camera pose
                camera_pose = np.eye(4)
                camera_pose = np.array([1.0, -1.0, -1.0, 1.0]).reshape(-1, 1) * camera_pose
                camera_pose = cam_tran @ camera_pose

                if self.vis_denoising:
                    ## generate smplx bodies in all denoising steps
                    ## visualize one body in all steps, visualize the denoising procedure
                    smplx_params = outputs[i, 0, ...] # <T, ...>
                    body_verts, body_faces, body_joints = dataloader.dataset.SMPLX.run(smplx_params, gender)
                    body_verts = body_verts.numpy()

                    save_path_gif = os.path.join(save_dir, f'{scene_id}', '000.gif')
                    save_imgs_dir = os.path.join(save_dir, f'{scene_id}', 'series')
                    timesteps = list(range(len(body_verts))) + [len(body_verts) - 1] * 10 # repeat last frame
                    for f, t in enumerate(timesteps):
                        meshes = {
                            'scenes': [scene_mesh], 
                            'bodies': [trimesh.Trimesh(vertices=body_verts[t], faces=body_faces)]
                        }
                        save_path = os.path.join(save_imgs_dir, f'{f:0>3d}.png')
                        render_prox_scene(meshes, camera_pose, save_path)
                    
                    ## convert images to gif
                    frame2gif(os.path.join(save_dir, f'{scene_id}', 'series'), save_path_gif, size=(480, 270))
                    os.system(f'rm -rf {save_imgs_dir}')
                else:
                    ## generate smplx bodies in last denoising step
                    ## only visualize the body in last step, but visualize multi bodies
                    smplx_params = outputs[i, :, -1, ...] # <k, ...>
                    body_verts, body_faces, body_joints = dataloader.dataset.SMPLX.run(smplx_params, gender)
                    body_verts = body_verts.numpy()

                    if self.save_mesh:
                        os.makedirs(os.path.join(save_dir, f'{scene_id}'), exist_ok=True)
                        scene_mesh.export(os.path.join(save_dir, f'{scene_id}', 'scene.ply'))

                        for j in range(len(body_verts)):
                            body_mesh = trimesh.Trimesh(vertices=body_verts[j], faces=body_faces)
                            ## render generated body separately
                            render_prox_scene({
                                'scenes': [scene_mesh],
                                'bodies': [body_mesh],
                            }, camera_pose, os.path.join(save_dir, f'{scene_id}', f'{j:0>3d}.png'))
                            ## save generated body mesh separately
                            body_mesh.export(os.path.join(save_dir, f'{scene_id}', f'body{j:0>3d}.ply'))
                    else:
                        meshes = {'scenes': [scene_mesh]}
                        meshes['bodies'] = []
                        for j in range(len(body_verts)):
                            meshes['bodies'].append(trimesh.Trimesh(vertices=body_verts[j], faces=body_faces))
                        save_path = os.path.join(save_dir, f'{scene_id}', '000.png')
                        render_prox_scene(meshes, camera_pose, save_path)
                
                cnt += 1
            
            if cnt >= self.vis_case_num:
                break

@VISUALIZER.register()
@torch.no_grad()
class MotionGenVisualizer():
    def __init__(self, cfg: DictConfig) -> None:
        """ Visual evaluation class for motion generation task.

        Args:
            cfg: visuzalizer configuration
        """
        self.vis_case_num = cfg.vis_case_num
        self.ksample = cfg.ksample
        self.vis_denoising = cfg.vis_denoising
        self.save_mesh = cfg.save_mesh
    
    def visualize(
        self, 
        model: torch.nn.Module, 
        dataloader: torch.utils.data.DataLoader, 
        save_dir: str,
    ) -> None:
        """ Visualize method

        Args:
            model: diffusion model
            dataloader: test dataloader
            save_dir: save directory of rendering images
        """
        model.eval()
        device = model.device

        cnt = 0
        for data in dataloader:
            for key in data:
                if torch.is_tensor(data[key]):
                    data[key] = data[key].to(device)
            data['normalizer'] = dataloader.dataset.normalizer
            data['repr_type'] = dataloader.dataset.repr_type
            
            ksample = 1 if self.vis_denoising else self.ksample
            outputs = model.sample(data, k=ksample) # <B, k, T, L, D>

            for i in range(outputs.shape[0]):
                scene_id = data['scene_id'][i]
                cam_tran = data['cam_tran'][i]
                gender = data['gender'][i]

                origin_cam_tran = data['origin_cam_tran'][i]
                scene_trans = cam_tran @ np.linalg.inv(origin_cam_tran) # scene_T @ origin_cam_T = cur_cam_T
                scene_mesh = dataloader.dataset.scene_meshes[scene_id].copy()
                scene_mesh.apply_transform(scene_trans)

                ## calculate camera pose
                camera_pose = np.eye(4)
                camera_pose = np.array([1.0, -1.0, -1.0, 1.0]).reshape(-1, 1) * camera_pose
                camera_pose = cam_tran @ camera_pose

                if self.vis_denoising:
                    ## generate smplx bodies in all denoising steps
                    ## visualize bodies in all steps, visualize the denoising procedure
                    smplx_params = outputs[i, 0, ...] # <T, ...>
                    body_verts, body_faces, body_joints = dataloader.dataset.SMPLX.run(smplx_params, gender)
                    body_verts = body_verts.numpy()

                    rand_str = random_str(4)
                    save_path_gif = os.path.join(save_dir, f'{scene_id}_{rand_str}', '000.gif')
                    save_imgs_dir = os.path.join(save_dir, f'{scene_id}_{rand_str}', 'series')
                    timesteps = list(range(len(body_verts))) + [len(body_verts) - 1] * 10 # repeat last frame
                    for f, t in enumerate(timesteps):
                        meshes = {
                            'scenes': [scene_mesh], 
                            'bodies': [trimesh.Trimesh(vertices=bv, faces=body_faces) for bv in body_verts[t]]
                        }
                        save_path = os.path.join(save_imgs_dir, f'{f:0>3d}.png')
                        render_prox_scene(meshes, camera_pose, save_path)
                    
                    ## convert images to gif
                    frame2gif(save_imgs_dir, save_path_gif, size=(480, 270))
                    os.system(f'rm -rf {save_imgs_dir}')
                else:
                    ## generate smplx bodies in all denoising step
                    ## only visualize the body in last step, visualize with gif
                    smplx_params = outputs[i, :, -1, ...] # <k, ...>
                    body_verts, body_faces, body_joints = dataloader.dataset.SMPLX.run(smplx_params, gender)
                    body_verts = body_verts.numpy()

                    rand_str = random_str(4)
                    if self.save_mesh:
                        os.makedirs(os.path.join(save_dir, f'{scene_id}_{rand_str}'), exist_ok=True)
                        scene_mesh.export(os.path.join(save_dir, f'{scene_id}_{rand_str}', 'scene.ply'))
                    
                    for k in range(len(body_verts)):
                        save_path_gif = os.path.join(save_dir, f'{scene_id}_{rand_str}', f'{k:3d}.gif')
                        save_imgs_dir = os.path.join(save_dir, f'{scene_id}_{rand_str}', 'series')
                        for j, body in enumerate(body_verts[k]):
                            body_mesh = trimesh.Trimesh(vertices=body, faces=body_faces)
                            meshes = {
                                'scenes': [scene_mesh],
                                'bodies': [body_mesh]
                            }
                            save_path = os.path.join(save_imgs_dir, f'{j:0>3d}.png')
                            render_prox_scene(meshes, camera_pose, save_path)

                            if self.save_mesh:
                                save_mesh_dir = os.path.join(save_dir, f'{scene_id}_{rand_str}', f'mesh{k:3d}')
                                os.makedirs(save_mesh_dir, exist_ok=True)
                                body_mesh.export(os.path.join(
                                    save_mesh_dir, f'body{j:0>3d}.obj'
                                ))
                        
                        ## convert image to gif
                        frame2gif(save_imgs_dir, save_path_gif, size=(480, 270))
                        os.system(f'rm -rf {save_imgs_dir}')

                cnt += 1
                if cnt >= self.vis_case_num:
                    break
            
            if cnt >= self.vis_case_num:
                break

@VISUALIZER.register()
@torch.no_grad()
class PathPlanningRenderingVisualizer():
    def __init__(self, cfg: DictConfig) -> None:
        """ Visual evaluation class for path planning task. Directly rendering images.

        Args:
            cfg: visuzalizer configuration
        """
        self.ksample = cfg.ksample
        self.vis_case_num = cfg.vis_case_num
        self.vis_denoising = cfg.vis_denoising
        self.scannet_mesh_dir = cfg.scannet_mesh_dir
    
    def visualize(
        self,
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        save_dir: str,
    ) -> None:
        """ Visualize method

        Args:
            model: diffusion model
            dataloader: test dataloader
            save_dir: save directory of rendering images
            vis_denoising: visualize denoising procedure, default is False
            vis_cnt: visualized sample count
        """
        model.eval()
        device = model.device

        cnt = 0
        for data in dataloader:
            for key in data:
                if torch.is_tensor(data[key]):
                    data[key] = data[key].to(device)
            data['normalizer'] = dataloader.dataset.normalizer
            data['repr_type'] = dataloader.dataset.repr_type
            
            ksample = 1 if self.vis_denoising else self.ksample
            outputs = model.sample(data, k=ksample) # <B, k, T, L, D>

            scene_id = data['scene_id']
            trans_mat = data['trans_mat']
            target = data['target'].cpu().numpy()
            for i in range(outputs.shape[0]):
                rand_str = random_str()

                ## load scene and camera pose
                scene_mesh = trimesh.load(os.path.join(
                    self.scannet_mesh_dir, 'mesh', f'{scene_id[i]}_vh_clean_2.ply'))
                scene_mesh.apply_transform(trans_mat[i])
                camera_pose = np.eye(4)
                camera_pose[0:3, -1] = np.array([0, 0, 7])
                
                ## save trajectory
                if self.vis_denoising:
                    save_path_gif = os.path.join(save_dir, f'{scene_id[i]}_{rand_str}', '000.gif')
                    save_imgs_dir = os.path.join(save_dir, f'{scene_id[i]}_{rand_str}', 'series')

                    sequences = outputs[i, 0, ...] # <T, horizon, 2>
                    timesteps = list(range(len(sequences))) + [len(sequences) - 1] * 10 # repeat last frame
                    for f, t in enumerate(timesteps):
                        path = sequences[t].cpu().numpy() # <horizon, 2>

                        render_scannet_path(
                            {'scene': scene_mesh, 
                            'target': create_trimesh_node(target[i], color=np.array([0, 255, 0], dtype=np.uint8)),
                            'path': create_trimesh_nodes_path(path, merge=True)},
                            camera_pose=camera_pose,
                            save_path=os.path.join(save_imgs_dir, f'{f:0>3d}.png')
                        )
                    frame2gif(save_imgs_dir, save_path_gif, size=(480, 270))
                    os.system(f'rm -rf {save_imgs_dir}')
                else:
                    save_imgs_dir = os.path.join(save_dir, f'{scene_id[i]}_{rand_str}')

                    sequences = outputs[i, :, -1, ...] # <k, horizon, 2>
                    for t in range(len(sequences)):
                        path = sequences[t].cpu().numpy() # <horizon, 2>

                        render_scannet_path(
                            {'scene': scene_mesh,
                            'target': create_trimesh_node(target[i], color=np.array([0, 255, 0], dtype=np.uint8)),
                            'path': create_trimesh_nodes_path(path, merge=True)},
                            camera_pose=camera_pose,
                            save_path=os.path.join(save_imgs_dir, f'{t:0>3d}.png')
                        )
                
                cnt += 1
                if cnt >= self.vis_case_num:
                    break
            
            if cnt >= self.vis_case_num:
                break

@VISUALIZER.register()
@torch.no_grad()
class GraspGenVisualizer():
    def __init__(self, cfg: DictConfig) -> None:
        """ Visual evaluation class for pose generation task.

        Args:
            cfg: visuzalizer configuration
        """
        self.ksample = cfg.ksample
        self.hand_model = get_handmodel(batch_size=self.ksample, device='cuda')

    def visualize(
            self,
            model: torch.nn.Module,
            dataloader: torch.utils.data.DataLoader,
            save_dir: str,
            vis_denoising: bool = False,
            vis_cnt: int = 20,
    ) -> None:
        """ Visualize method

        Args:
            model: diffusion model
            dataloader: test dataloader
            save_dir: save directory of rendering images
            vis_denoising: visualize denoising procedure, default is False
            vis_cnt: visualized sample count
        """
        model.eval()
        device = model.device

        cnt = 0
        ksample = 1 if vis_denoising else self.ksample
        assert (vis_denoising is False)

        os.makedirs(save_dir, exist_ok=True)
        for data in dataloader:
            for key in data:
                if torch.is_tensor(data[key]):
                    data[key] = data[key].to(device)

            outputs = model.sample(data, k=ksample)  # B x ksample x n_steps x 33
            for i in range(outputs.shape[0]):
                scene_id = data['scene_id'][i]
                scene_dataset, scene_object = scene_id.split('+')
                mesh_path = os.path.join('assets/object', scene_dataset, scene_object, f'{scene_object}.stl')
                obj_mesh = trimesh.load(mesh_path)

                hand_qpos = outputs[i, :, -1, ...]
                self.hand_model.update_kinematics(q=hand_qpos)
                for j in range(ksample):
                    vis_data = [plot_mesh(obj_mesh, color='lightblue')]
                    vis_data += self.hand_model.get_plotly_data(i=j, opacity=0.8, color='pink')
                    save_path = os.path.join(save_dir, f'{scene_id}+sample-{j}.html')
                    fig = go.Figure(data=vis_data)
                    fig.write_html(save_path)
                cnt += 1
                if cnt >= vis_cnt:
                    break

@VISUALIZER.register()
# @torch.no_grad()
class GraspGenURVisualizer():
    def __init__(self, cfg: DictConfig) -> None:
        """ Visual evaluation class for pose generation task.
        Args:
            cfg: visuzalizer configuration
        """
        self.ksample = cfg.ksample
        self.hand_model = get_handmodel(batch_size=1, device='cuda')
        self.use_llm = cfg.use_llm
        # 加载average_scales.pkl文件
        ###dexgrasp###
        self.average_scales = self.load_average_scales('/inspurfs/group/mayuexin/datasets/DexGraspNet/scales.pkl')
        ###unidex####
        # self.average_scales = self.load_average_scales( '/inspurfs/group/mayuexin/datasets/UniDexGrasp/DFCData/scales.pkl')

        # Getting descriptions from LLM
        if self.use_llm:
            # with open("/inspurfs/group/mayuexin/datasets/dex_hand_preprocessed/UniDexGrasp_gpt4o_mini.json", "r") as jsonfile:
            # with open("/inspurfs/group/mayuexin/datasets/dex_hand_preprocessed/DexGraspnet_gpt4o_mini.json", "r") as jsonfile:
            # with open("/inspurfs/group/mayuexin/datasets/dex_hand_preprocessed/DexGRAB_gpt4o_mini.json", "r") as jsonfile:
            # with open("/inspurfs/group/mayuexin/datasets/dex_hand_preprocessed/Realdex_gpt4o_mini.json", "r") as jsonfile:
            with open("/inspurfs/group/mayuexin/datasets/dex_hand_preprocessed/multidex_gpt4o_mini.json", "r") as jsonfile:
            # with open("/inspurfs/group/mayuexin/zym/diffusion+hand/graps_gen/multidex_gpt4o_mini.json", "r") as jsonfile:
                self.scene_text = json.load(jsonfile)
            # pre-process for tokenizer
            for k, text in self.scene_text.items():
                txtclips = text.split("\n")
                self.scene_text[k] = txtclips[:]
            # for k, text in self.scene_text.items():
            #     txtclips = text.split("\n")
            #     if len(txtclips) <= 3: 
            #         txtclips = text.split(".")
            #         self.scene_text[k] = txtclips
            #     else:
            #         self.scene_text[k] = txtclips[2:-2]

    def load_average_scales(self, file_path):
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    def visualize(
            self,
            cfg,
            model: torch.nn.Module,
            dataloader: torch.utils.data.DataLoader,
            save_dir: str,
            evaluation_grasp: None,
            evaluation_grasp_all: None
    ) -> None:
        """ Visualize method
        Args:
            model: diffusion model
            dataloader: test dataloader
            save_dir: save directory of rendering images
        """
        # model.eval()
        device = model.device
        autocast = contextlib.nullcontext

        # dataset = torch.load(os.path.join(cfg.dataset_file_path))
        # transformation_vec = dataset["metadata"][0]['translations']
        save_dir = '/inspurfs/group/mayuexin/zhuyufei/graps_gen/outputs/visualize_outputs'
        sub_total_epoches = 10
        os.makedirs(cfg.save_RL_dir, exist_ok=True)
        # save
        pbar = tqdm(total=len(dataloader.dataset.split) * self.ksample)
        object_pcds_dict = dataloader.dataset.scene_pcds
        res = {'method': 'diffuser@w/o-opt',
               'desc': 'w/o optimizer grasp pose generation',
               'sample_qpos': {}}

        # for syn training dataset
        # train_obj_list_ds = ["contactdb+alarm_clock", "contactdb+banana", "contactdb+binoculars",
        #                      "contactdb+cube_medium", "contactdb+mouse", "contactdb+piggy_bank",
        #                      "contactdb+ps_controller", "contactdb+stapler", "contactdb+train",
        #                      "ycb+toy_airplane"]

        # 从 dataloader.dataset._test_split 随机选择 100 个物体
        # for object_name in train_obj_list_ds:
        
        #dexgrasp 
        # selected_objects = random.sample(dataloader.dataset._test_split, 100)
        # for object_name in selected_objects:  # slected 100 obj
        optimizer_cls = torch.optim.Adam
        params = []
        for n, p in model.named_parameters():
            if p.requires_grad:
                params.append(p)
            # nparams.append(p.nelement())
        
        optimizer = optimizer_cls(
            params,
            lr=cfg.learning_rate,
            betas=(cfg.adam_beta1, cfg.adam_beta2),
            weight_decay=cfg.adam_weight_decay,
            eps=cfg.adam_epsilon,
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.1)
        global_step = 0
        
        # disc = DiscriminatorTrainer.load_from_checkpoint(cfg.MODEL.discriminator.disc.discriminator_checkpoint, cfg=cfg)
        # discriminator = disc.discriminator.to(device)
        # discriminator.eval()
        # train
        ref = copy.deepcopy(model.eps_model)
        model.eps_model.train()
        # info = defaultdict(list)
        beta = cfg.beta
        eps = cfg.eps
        
        all_epoch_data = {
            'succ6s': [],
            'succ1s': [],
            'collision_values': [],
            'collision_values2': []
        }
        total_epoches = 20
        for epoch_ in range(0, total_epoches):
            succ6s = []
            succ1s = []
            collision = []
            collision2 = []
            for object_name in dataloader.dataset._test_split:    # use all obj
                model.eval()
                global_step += 1
                
                scale = self.average_scales.get(object_name, 1.0)  # 获取物体的平均scale，默认为1.0
                obj_pcd_can = torch.tensor(object_pcds_dict[object_name], device=device).unsqueeze(0).repeat(self.ksample, 1, 1) * scale
                obj_pcd_nor = obj_pcd_can[:, :dataloader.dataset.num_points, 3:]
                obj_pcd_can = obj_pcd_can[:, :dataloader.dataset.num_points, :3]

                i_rot_list = []
                for k_rot in range(self.ksample):
                    i_rot_list.append(random_rot(device))

                # i_rot = random_rot(device).repeat(self.ksample, 1, 1).to(torch.float64).to(device)
                i_rot = torch.stack(i_rot_list)
                # pdb.set_trace()
                if cfg.task.name == 'grasp_gen_ur':
                    obj_pcd_rot = torch.matmul(i_rot, obj_pcd_can.transpose(1, 2)).transpose(1, 2)
                    obj_pcd_nor_rot = torch.matmul(i_rot, obj_pcd_nor.transpose(1, 2)).transpose(1, 2)
                else:
                    obj_pcd_rot = obj_pcd_can
                    obj_pcd_nor_rot = obj_pcd_nor
                
                all_sentence = []
                for n in range(self.ksample):
                    if self.use_llm:
                        all_sentence.extend(self.scene_text[object_name])        
                
                # construct data
                data = {'x': torch.randn(self.ksample, 27, device=device) if cfg.task.name == 'grasp_gen_ur' else torch.randn(self.ksample, 33, device=device),
                            'pos': obj_pcd_rot.to(device),
                            'normal':obj_pcd_nor_rot.to(device),
                            'feat':obj_pcd_nor_rot.to(device),
                            'scene_rot_mat': i_rot,
                            'scene_id': [object_name for i in range(self.ksample)],
                            'cam_trans': [None for i in range(self.ksample)],
                            'text': (all_sentence if self.use_llm else None),
                            'sentence_cnt': ([len(self.scene_text[object_name])] * self.ksample if self.use_llm else None)}

                    ## 'TODO:USE Transformer 'squeeze the first dimension of pos and feat
                offset, count = [], 0
                for item in data['pos']:
                    count += item.shape[0]
                    offset.append(count)
                offset = torch.IntTensor(offset)
                data['offset'] = offset.to(device)
                data['pos'] = rearrange(data['pos'], 'b n c -> (b n) c').to(device)
                data['feat'] = rearrange(data['feat'], 'b n c -> (b n) c').to(device)
                # sample
                # visualization for checking
                scene_id = data['scene_id'][0]

                #####   multidex
                scene_dataset, scene_object = scene_id.split('+')
                mesh_path = os.path.join('assets/object', scene_dataset, scene_object, f'{scene_object}.stl')
                obj_mesh = trimesh.load(mesh_path)

                # samples_pc, fid = obj_mesh.sample(cfg.pc_num_points, return_index=True)
                # object_pc = torch.tensor(samples_pc, dtype=torch.float).unsqueeze(0).repeat(outputs.size(0), 1, 1).to(device)
                
                ###   realdex
                # scene_object = scene_id
                # mesh_path = os.path.join('/inspurfs/group/mayuexin/datasets/Realdex/meshdata', f'{scene_object}.obj')
                # obj_mesh = trimesh.load(mesh_path)

                ##    dexgrab
                # scene_object = scene_id
                # mesh_path = os.path.join('/inspurfs/group/mayuexin/datasets/DexGRAB/contact_meshes', f'{scene_object}.ply')
                # obj_mesh = trimesh.load(mesh_path)

                ##### dexgrasp
                # scene_object = scene_id
                # mesh_path = os.path.join('/inspurfs/group/mayuexin/datasets/DexGraspNet/meshdata', scene_object,  'coacd','decomposed.obj')
                # obj_mesh = trimesh.load(mesh_path)
                # obj_mesh.vertices *= scale

                samples = []
                save_dir_end = os.path.join(save_dir, f'{object_name}', f'{epoch_}')
                os.makedirs(save_dir_end, exist_ok=True)

                latents, log_probs, timesteps = model.sample(data, k=1)
                # timesteps = timesteps.repeat(32, 1)
                outputs = latents.squeeze(1)[-1, :].to(torch.float64)
                latents = latents[:, -1, :].to(torch.float64).unsqueeze(0)

                # log_probs = log_probs[:, -1, :].to(torch.float64)

                ## denormalization
                if cfg.task.name == 'grasp_gen_ur':
                    if dataloader.dataset.normalize_x:
                        outputs[:, 3:] = dataloader.dataset.angle_denormalize(joint_angle=outputs[:, 3:].cpu()).cuda()
                    if dataloader.dataset.normalize_x_trans:
                        outputs[:, :3] = dataloader.dataset.trans_denormalize(global_trans=outputs[:, :3].cpu()).cuda()
                else:
                    if dataloader.dataset.normalize_x:
                        outputs[:, 9:] = dataloader.dataset.angle_denormalize(joint_angle=outputs[:, 9:].cpu()).cuda()
                    if dataloader.dataset.normalize_x_trans:
                        outputs[:, :3] = dataloader.dataset.trans_denormalize(global_trans=outputs[:, :3].cpu()).cuda()
                
                if cfg.task.name == 'grasp_gen_ur':
                    id_6d_rot = torch.tensor([1., 0., 0., 0., 1., 0.], device=device).view(1, 6).repeat(self.ksample, 1).to(torch.float64)
                    outputs_3d_rot = rot_to_orthod6d(torch.bmm(i_rot.transpose(1, 2), robust_compute_rotation_matrix_from_ortho6d(id_6d_rot)))
                    outputs[:, :3] = torch.bmm(i_rot.transpose(1, 2), outputs[:, :3].unsqueeze(-1)).squeeze(-1)
                    batch_size = outputs.size(0)
                    # 如果 outputs[:, 3:] 是 b*22 的话，添加两个 0 列
                    if outputs[:, 3:].size(1) == 22:
                        zeros = torch.zeros((batch_size, 2), device=outputs.device)
                        # 拼接操作
                        outputs = torch.cat([outputs[:, :3], outputs_3d_rot, zeros, outputs[:, 3:]], dim=-1)
                    else:
                        # 如果 outputs[:, 3:] 是 b*24 的话，不做任何处理
                        outputs = torch.cat([outputs[:, :3], outputs_3d_rot, outputs[:, 3:]], dim=-1)

                for i in range(outputs.shape[0]):
                    '''
                    self.hand_model.update_kinematics(q=outputs[i:i+1, :])
                    vis_data = [plot_mesh(obj_mesh, color='lightpink')]
                    vis_data += self.hand_model.get_plotly_data(opacity=1.0, color='#8799C6')
                    if i < 100:
                        # 保存为 HTML 文件
                        save_path = os.path.join(save_dir_end, f'{object_name}_sample-{i}.html')
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
                    pbar.update(1)
                res['sample_qpos'][object_name] = np.array(outputs.cpu().detach())

                # compute rewards using discriminator  or issac gym
                '''
                while True:
                    try:
                        selected_indices = input(f"Enter the indices that align with human preferences {object_name} (e.g., 0 1 6 19): ")
                        selected_indices = list(map(int, selected_indices.split()))
                        break
                    except ValueError:
                        print("Invalid input. Please enter space-separated numbers.")
                '''
                
                _, collision_values, collision_values2, rewards_, rewards1_ = evaluation_grasp(cfg=cfg, grasps=res, obj_mesh=obj_mesh, device=device, object_name=object_name)
                # pdb.set_trace()
                succ6 = (rewards_.sum().item() + len(rewards_)) / (len(rewards_) * 2)
                succ1 = (rewards1_.sum().item() + len(rewards1_)) / (len(rewards1_) * 2)
                succ6s.append(succ6)
                succ1s.append(succ1)
                collision.append(collision_values)
                collision2.append(collision_values2)
                # rewards = torch.where(rewards_, torch.tensor(1, device='cuda:0'), torch.tensor(-1, device='cuda:0'))
                rewards = rewards_
                
                # evaluations.append(rewards)
                # output_tensor = -1 * torch.ones(32)
                # output_tensor[selected_indices] = 1
                # rewards = output_tensor.to(device)

                current_latents = latents[:, :-1, :].to(torch.float32)
                next_latents = latents[:, 1:, :].to(torch.float32)

                samples.append(
                        {
                            "timesteps": timesteps,
                            "latents": current_latents,  # each entry is the latent before timestep t
                            "next_latents": next_latents,  # each entry is the latent after timestep t
                            # "log_probs": log_probs.to(torch.float32),
                            "poses":outputs.to(torch.float32),
                            "rewards":torch.as_tensor(rewards, device=device),
                        }
                )
                
                samples = {k: torch.cat([s[k] for s in samples]) for k in samples[0].keys()}
                poses = samples["poses"]
            
                # save prompts
                del samples["poses"]
                torch.cuda.empty_cache()
                total_batch_size, num_timesteps = samples["timesteps"].shape
                # assert total_batch_size == config.sample.batch_size * config.sample.num_batches_per_epoch
                # assert num_timesteps == config.sample.num_steps
                # orig_sample = copy.deepcopy(samples)
                #################### TRAINING ####################
            #     num_inner_epochs = 1
            # for inner_epoch in range(num_inner_epochs):
                # shuffle samples along batch dimension
                perm = torch.randperm(total_batch_size, device=device)
                # samples = {k: v[perm] for k, v in orig_sample.items()}
                samples_1 = {k: v[perm] for k, v in samples.items()}
                samples = samples_1

                # shuffle along time dimension independently for each sample
                perms = torch.stack(
                    [torch.randperm(num_timesteps, device=device) for _ in range(total_batch_size)]
                )
                for key in ["latents", "next_latents"]:
                    samples[key] = samples[key][torch.arange(total_batch_size, device=device)[:, None], perms]
                    # samples[key] = tmp
                samples["timesteps"] = samples["timesteps"][torch.arange(total_batch_size, device=device)[:, None], perms].unsqueeze(1) # .repeat(1,2,1)
                # samples["log_probs"] = samples["log_probs"][torch.arange(total_batch_size, device=device)[:, None], perms]
                samples["timesteps"] = samples["timesteps"].repeat(1, len(samples['latents'][0, 0]), 1).permute(0, 2, 1)
                # rebatch for training
                # samples_batched = {k: v.reshape(-1, config.train.batch_size, *v.shape[1:]) for k, v in samples.items()}       #####
                # dict of lists -> list of dicts for easier iteration
                # samples_batched = [dict(zip(samples_batched, x)) for x in zip(*samples_batched.values())]                     #####
                ref = copy.deepcopy(model.eps_model)
                model.eps_model.train()
                
                # pdb.set_trace()
                for _ in range(0, sub_total_epoches):
                    for epoch in tqdm(range(0,total_batch_size),
                            desc="Update",
                            position=2,
                            leave=False, 
                            ):

                        sample_0 = {}
                        # sample_1 = {}
                        for key, value in samples.items():
                            sample_0[key] = value[epoch]

                        # sample_1[key] = value[i:i+config.train.batch_size, 1]
                        '''
                        if config.train.cfg:
                            # concat negative prompts to sample prompts to avoid two forward passes
                            embeds_0 = torch.cat([train_neg_prompt_embeds, sample_0["prompt_embeds"]])
                            embeds_1 = torch.cat([train_neg_prompt_embeds, sample_1["prompt_embeds"]])
                        else:
                            embeds_0 = sample_0["prompt_embeds"]
                            embeds_1 = sample_1["prompt_embeds"]
                        '''
                        losses = []
                        for j in tqdm(
                            range(num_timesteps),
                            desc="Timestep",
                            position=3,
                            leave=False,
                            # disable=not accelerator.is_local_main_process,
                        ):  
                            # with accelerator.accumulate(model.eps_model):
                            with autocast():
                                noise_pred_0 = model.eps_model(
                                        sample_0["latents"][j],
                                        sample_0["timesteps"][j],
                                        data['cond'],
                                )
                                noise_ref_pred_0 = ref(
                                        sample_0["latents"][j],
                                        sample_0["timesteps"][j],
                                        data['cond'],
                                )
                                # compute the log prob of next_latents given latents under the current model
                                total_prob_0 = model.sample_logprob_CM(                                #############
                                    noise_pred_0,
                                    sample_0["timesteps"][j],
                                    sample_0["latents"][j],
                                    # eta=model.eta,
                                    prev_sample=sample_0["next_latents"][j],
                                )
                                total_ref_prob_0 = model.sample_logprob_CM(                            #############
                                    noise_ref_pred_0,
                                    sample_0["timesteps"][j],
                                    sample_0["latents"][j],
                                    # eta=model.eta,
                                    prev_sample=sample_0["next_latents"][j],
                                )
                                # human_prefer = compare(sample_0['rewards'],sample_1['rewards'])
                                # clip the Q value
                                ratio = {}
                                tmp1 = 0
                                tmp2 = 0
                                num_pos = 0
                                for i in range(len(total_ref_prob_0)):
                                    ratio[i] = torch.clamp(torch.exp((total_prob_0[i]-total_ref_prob_0[i]).clamp(max=60)),1 - eps, 1 + eps)
                                # ratio_1 = torch.clamp(torch.exp(total_prob_1-total_ref_prob_1),1 - config.train.eps, 1 + config.train.eps)
                                    if rewards[i] == 1:
                                        tmp1 += beta*(torch.log(ratio[i]))*rewards[i]
                                        num_pos = num_pos + 1
                                    else:
                                        tmp2 += beta*(torch.log(ratio[i]))*rewards[i]
                                tmp = tmp1 + tmp2
                                # tmp = tmp1 / (num_pos + 1) + tmp2 / (len(total_ref_prob_0) + 1 - num_pos)
                                loss = -torch.log(torch.sigmoid(tmp)).mean()
                                losses.append(loss.item())
                                # backward pass
                                loss.backward(retain_graph=True)#
                                optimizer.step()
                                scheduler.step()
                                optimizer.zero_grad()
                        print(np.mean(losses))
            logger.info(f'**all 6dir Success Rate** across all objects: {np.mean(succ6s)}')
            logger.info(f'**one 6dir Success Rate** across all objects: {np.mean(succ1s)}')
            logger.info(f'**Collision** (depth: mm.) across succ grasps: {np.mean(collision_values) * 1e3}')
            logger.info(f'**Collision** (depth: mm.) across all grasps: {np.mean(collision_values2) * 1e3}')
            all_epoch_data['succ6s'].append(succ6s)
            all_epoch_data['collision_values'].append(collision_values)
            all_epoch_data['collision_values2'].append(collision_values2)
            # rewards_all = evaluation_grasp_all(cfg=cfg, grasps=res, device=device)
            # '''
            path = os.path.join(
                cfg.save_RL_dir,
                f'model_RL_{epoch_}.pth'
            )
            save_ckpt(
                model=model, epoch=global_step, step=global_step, path=path,
                save_scene_model=cfg.save_scene_model,
            )
            # '''
        # torch.save(all_epoch_data, '/inspurfs/group/mayuexin/zhuyufei/graps_gen/outputs/all_epoch_data/all_epoch_data.pt')
        # Checks if the accelerator has performed an optimization step behind the scenes

@VISUALIZER.register()
@torch.no_grad()
class PoseGenVisualizerHF():
    def __init__(self, cfg: DictConfig) -> None:
        """ Visual evaluation class for pose generation task.

        Args:
            cfg: visuzalizer configuration
        """
        self.ksample = cfg.ksample

    def visualize(
        self, 
        model: torch.nn.Module, 
        dataloader: torch.utils.data.DataLoader, 
    ) -> Any:
        """ Visualize method

        Args:
            model: diffusion model
            dataloader: test dataloader
        
        Return:
            Results for gradio rendering.
        """
        model.eval()
        device = model.device

        for data in dataloader:
            for key in data:
                if torch.is_tensor(data[key]):
                    data[key] = data[key].to(device)
            data['normalizer'] = dataloader.dataset.normalizer
            
            outputs = model.sample(data, k=self.ksample) # <B, k, T, D>
            
            i = 0
            scene_id = data['scene_id'][i]
            cam_tran = data['cam_tran'][i]
            gender = data['gender'][i]
            
            origin_cam_tran = data['origin_cam_tran'][i]
            scene_trans = cam_tran @ np.linalg.inv(origin_cam_tran) # scene_T @ origin_cam_T = cur_cam_T
            scene_mesh = dataloader.dataset.scene_meshes[scene_id].copy()
            scene_mesh.apply_transform(scene_trans)

            ## calculate camera pose
            camera_pose = np.eye(4)
            camera_pose = np.array([1.0, -1.0, -1.0, 1.0]).reshape(-1, 1) * camera_pose
            camera_pose = cam_tran @ camera_pose

            ## generate smplx bodies in last denoising step
            ## only visualize the body in last step, but visualize multi bodies
            smplx_params = outputs[i, :, -1, ...] # <k, ...>
            body_verts, body_faces, body_joints = dataloader.dataset.SMPLX.run(smplx_params, gender)
            body_verts = body_verts.numpy()

            res_images = []
            for j in range(len(body_verts)):
                body_mesh = trimesh.Trimesh(vertices=body_verts[j], faces=body_faces)
                ## render generated body separately
                img = render_prox_scene({'scenes': [scene_mesh], 'bodies': [body_mesh]}, camera_pose, None)
                res_images.append(img)
            return res_images

@VISUALIZER.register()
@torch.no_grad()
class MotionGenVisualizerHF():
    def __init__(self, cfg: DictConfig) -> None:
        """ Visual evaluation class for motion generation task.

        Args:
            cfg: visuzalizer configuration
        """
        self.ksample = cfg.ksample
    
    def visualize(
        self, 
        model: torch.nn.Module, 
        dataloader: torch.utils.data.DataLoader, 
    ) -> None:
        """ Visualize method

        Args:
            model: diffusion model
            dataloader: test dataloader
        
        Return:
            Results for gradio rendering.
        """
        model.eval()
        device = model.device

        for data in dataloader:
            for key in data:
                if torch.is_tensor(data[key]):
                    data[key] = data[key].to(device)
            data['normalizer'] = dataloader.dataset.normalizer
            data['repr_type'] = dataloader.dataset.repr_type
            
            outputs = model.sample(data, k=self.ksample) # <B, k, T, L, D>

            i = 0
            scene_id = data['scene_id'][i]
            cam_tran = data['cam_tran'][i]
            gender = data['gender'][i]

            origin_cam_tran = data['origin_cam_tran'][i]
            scene_trans = cam_tran @ np.linalg.inv(origin_cam_tran) # scene_T @ origin_cam_T = cur_cam_T
            scene_mesh = dataloader.dataset.scene_meshes[scene_id].copy()
            scene_mesh.apply_transform(scene_trans)

            ## calculate camera pose
            camera_pose = np.eye(4)
            camera_pose = np.array([1.0, -1.0, -1.0, 1.0]).reshape(-1, 1) * camera_pose
            camera_pose = cam_tran @ camera_pose

            ## generate smplx bodies in all denoising step
            ## only visualize the body in last step, visualize with gif
            smplx_params = outputs[i, :, -1, ...] # <k, ...>
            body_verts, body_faces, body_joints = dataloader.dataset.SMPLX.run(smplx_params, gender)
            body_verts = body_verts.numpy()
            
            res_ksamples = []
            for k in range(len(body_verts)):
                res_images = []
                for j, body in enumerate(body_verts[k]):
                    body_mesh = trimesh.Trimesh(vertices=body, faces=body_faces)
                    img = render_prox_scene({'scenes': [scene_mesh], 'bodies': [body_mesh]}, camera_pose, None)
                    res_images.append(img)
                res_ksamples.append(res_images)
            return res_ksamples

@VISUALIZER.register()
@torch.no_grad()
class PathPlanningRenderingVisualizerHF():
    def __init__(self, cfg: DictConfig) -> None:
        """ Visual evaluation class for path planning task. Directly rendering images.

        Args:
            cfg: visuzalizer configuration
        """
        self.ksample = cfg.ksample
        self.scannet_mesh_dir = cfg.scannet_mesh_dir
    
    def visualize(
        self,
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
    ) -> None:
        """ Visualize method

        Args:
            model: diffusion model
            dataloader: test dataloader
        """
        model.eval()
        device = model.device

        for data in dataloader:
            for key in data:
                if torch.is_tensor(data[key]):
                    data[key] = data[key].to(device)
            data['normalizer'] = dataloader.dataset.normalizer
            data['repr_type'] = dataloader.dataset.repr_type
            
            outputs = model.sample(data, k=self.ksample) # <B, k, T, L, D>

            scene_id = data['scene_id']
            trans_mat = data['trans_mat']
            target = data['target'].cpu().numpy()
            i = 0

            ## load scene and camera pose
            scene_mesh = trimesh.load(os.path.join(
                self.scannet_mesh_dir, 'mesh', f'{scene_id[i]}_vh_clean_2.ply'))
            scene_mesh.apply_transform(trans_mat[i])
            camera_pose = np.eye(4)
            camera_pose[0:3, -1] = np.array([0, 0, 10])

            sequences = outputs[i, :, -1, ...] # <k, horizon, 2>
            res_images = []
            for t in range(len(sequences)):
                path = sequences[t].cpu().numpy() # <horizon, 2>

                img = render_scannet_path(
                    {'scene': scene_mesh,
                    'target': create_trimesh_node(target[i], color=np.array([0, 255, 0], dtype=np.uint8)),
                    'path': create_trimesh_nodes_path(path, merge=True)},
                    camera_pose=camera_pose,
                    save_path=None
                )
                res_images.append(img)
            
            return res_images

def create_visualizer(cfg: DictConfig) -> nn.Module:
    """ Create a visualizer for visual evaluation
    Args:
        cfg: configuration object
        slurm: on slurm platform or not. This field is used to specify the data path
    
    Return:
        A visualizer
    """
    return VISUALIZER.get(cfg.name)(cfg)

