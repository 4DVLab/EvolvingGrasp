import os
import json
import torch
import torch.nn as nn
import numpy as np
from einops import rearrange
import trimesh
import pickle
from omegaconf import DictConfig
from plotly import graph_objects as go
from typing import Any
import random, pdb
from utils.misc import random_str
from utils.registry import Registry
from utils.visualize import frame2gif, render_prox_scene, render_scannet_path
from utils.visualize import create_trimesh_nodes_path, create_trimesh_node
from utils.handmodel import get_handmodel
from utils.plotly_utils import plot_mesh
from utils.rot6d import rot_to_orthod6d, robust_compute_rotation_matrix_from_ortho6d, random_rot
from tqdm import tqdm
from ours_utils.hand_transformation import apply_transformations
import time
from loguru import logger

VISUALIZER = Registry('Visualizer')

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
@torch.no_grad()
class GraspGenURVisualizer():
    def __init__(self, cfg: DictConfig) -> None:
        """ Visual evaluation class for pose generation task.
        Args:
            cfg: visuzalizer configuration
        """
        self.visualize_html = cfg.visualize_html
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
            # diffuser: torch.nn.Module,
            dataloader: torch.utils.data.DataLoader,
            save_dir: str,
            evaluation_grasp_all: None,
            evaluation_grasp: None
    ) -> None:
        """ Visualize method
        Args:
            model: diffusion model
            dataloader: test dataloader
            save_dir: save directory of rendering images
        """
        model.eval()
        device = model.device

        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'html'), exist_ok=True)
        # save
        if self.visualize_html:
            pbar = tqdm(total=len(dataloader.dataset.split) * self.ksample)
        object_pcds_dict = dataloader.dataset.scene_pcds
        res = {'method': 'EvolvingGrasp@w/o-opt',
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
        cost_time = []
        # epoch_obj_i = 2
        for object_name in dataloader.dataset._test_split:    # use all obj
            scale = self.average_scales.get(object_name, 1.0)  # 获取物体的平均scale，默认为1.0
            obj_pcd_can = torch.tensor(object_pcds_dict[object_name], device=device).unsqueeze(0).repeat(self.ksample, 1, 1) * scale
            # pdb.set_trace()
            #### unidex
            # selected_objects = random.sample(dataloader.dataset._test_split, 100)
            # for object_name in selected_objects:
            #     scale = self.average_scales.get(object_name, 1.0)  # 获取物体的平均scale，默认为1.0
            #     obj_pcd_can = torch.tensor(object_pcds_dict[object_name], device=device).unsqueeze(0).repeat(self.ksample, 1, 1) / scale
            ###scale = 1
            # for object_name in dataloader.dataset._test_split:
            #     obj_pcd_can = torch.tensor(object_pcds_dict[object_name], device=device).unsqueeze(0).repeat(self.ksample, 1, 1)
            
            obj_pcd_nor = obj_pcd_can[:, :dataloader.dataset.num_points, 3:]
            obj_pcd_can = obj_pcd_can[:, :dataloader.dataset.num_points, :3]
            # import open3d as o3d
            # import numpy as np

            # # Assuming obj_pcd_can is your tensor containing the point cloud data
            # # Convert the tensor to a numpy array
            # obj_pcd_can_np = obj_pcd_can.cpu().numpy()

            # # Create an Open3D PointCloud object
            # pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(obj_pcd_can_np[0])

            # # Optionally, if you have normals and want to save them
            # if obj_pcd_nor is not None:
            #     obj_pcd_nor_np = obj_pcd_nor.cpu().numpy()
            #     pcd.normals = o3d.utility.Vector3dVector(obj_pcd_nor_np[0])

            # # Save the point cloud as a PLY file
            # o3d.io.write_point_cloud(f"{object_name}.ply", pcd)
            i_rot_list = []
            for k_rot in range(self.ksample):
                i_rot_list.append(random_rot(device))
            i_rot = torch.stack(i_rot_list).to(torch.float64)
            # i_rot = random_rot(device).repeat(self.ksample, 1, 1).to(torch.float64).to(device)
            if cfg.task.name == 'grasp_gen_ur':
                obj_pcd_rot = torch.matmul(i_rot, obj_pcd_can.transpose(1, 2)).transpose(1, 2)
                obj_pcd_nor_rot = torch.matmul(i_rot, obj_pcd_nor.transpose(1, 2)).transpose(1, 2)
            else:
                obj_pcd_rot = obj_pcd_can
                obj_pcd_nor_rot = obj_pcd_nor
            
                
            # construct data
            data = {'x': torch.randn(self.ksample, 33, device=device),
                    'pos': obj_pcd_rot.to(device),
                    'normal':obj_pcd_nor_rot.to(device),
                    'feat':obj_pcd_nor_rot.to(device),
                    'scene_rot_mat': i_rot,
                    'scene_id': [object_name for i in range(self.ksample)],
                    'cam_trans': [None for i in range(self.ksample)],
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
            start_time = time.time()
            outputs, _, _ = model.sample(data, k=1)
            end_time = time.time()
            cost_time.append(end_time - start_time)
            # logger.info(f'The inference time is: {end_time - start_time}')
            # pdb.set_trace()

            outputs = outputs.squeeze(1)[-1, :].to(torch.float64)
            # print('outputs.shape:',outputs.shape)
            # print('outputs:',outputs)
            ## denormalization
            if dataloader.dataset.normalize_x_trans:
                outputs[:, :3] = dataloader.dataset.trans_denormalize(global_trans=outputs[:, :3].cpu()).cuda()

            # if cfg.task.name == 'grasp_gen_ur':
            #     if dataloader.dataset.normalize_x:
            #         outputs[:, 3:] = dataloader.dataset.angle_denormalize(joint_angle=outputs[:, 3:].cpu()).cuda()
            # else:
            if dataloader.dataset.normalize_x:
                outputs[:, 9:] = dataloader.dataset.angle_denormalize(joint_angle=outputs[:, 9:].cpu()).cuda()
            batch_size = outputs.size(0)
            '''
            if cfg.task.name == 'grasp_gen_ur':
                id_6d_rot = torch.tensor([1., 0., 0., 0., 1., 0.], device=device).view(1, 6).repeat(self.ksample, 1).to(torch.float64)
                outputs_3d_rot = rot_to_orthod6d(torch.bmm(i_rot.transpose(1, 2), robust_compute_rotation_matrix_from_ortho6d(id_6d_rot)))
                outputs[:, :3] = torch.bmm(i_rot.transpose(1, 2), outputs[:, :3].unsqueeze(-1)).squeeze(-1)
                # 如果 outputs[:, 3:] 是 b*22 的话，添加两个 0 列
                # outputs[:, 3:] = apply_transformations(outputs[:, 3:], device,
                #                  robust_compute_rotation_matrix_from_ortho6d(id_6d_rot).to(torch.float32), outputs[:, :3].to(torch.float32))
                # pdb.set_trace()
                if outputs[:, 3:].size(1) == 22:
                    zeros = torch.zeros((batch_size, 2), device=outputs.device)
                    # 拼接操作
                    outputs = torch.cat([outputs[:, :3], outputs_3d_rot, zeros, outputs[:, 3:]], dim=-1)
                else:
                    # 如果 outputs[:, 3:] 是 b*24 的话，不做任何处理
                    outputs = torch.cat([outputs[:, :3], outputs_3d_rot, outputs[:, 3:]], dim=-1)
            '''
            # else:
                # pdb.set_trace()
                # outputs[:, :3] = torch.bmm(robust_compute_rotation_matrix_from_ortho6d(outputs[:, 3:9]).transpose(1, 2), outputs[:, :3].unsqueeze(-1)).squeeze(-1)
            # visualization for checking
            scene_id = data['scene_id'][0]

            ##### dexGRAB
            # scene_object = scene_id
            # mesh_path = os.path.join('/inspurfs/group/mayuexin/datasets/DexGRAB/contact_meshes', f'{scene_object}.ply')
            # obj_mesh = trimesh.load(mesh_path)

            #####   multidex
            if dataloader.dataset.datasetname == 'MultiDexShadowHandUR':
                scene_dataset, scene_object = scene_id.split('+')
                mesh_path = os.path.join('assets/object', scene_dataset, scene_object, f'{scene_object}.stl')
                obj_mesh = trimesh.load(mesh_path)

            ###   realdex
            # scene_object = scene_id
            # mesh_path = os.path.join('/inspurfs/group/mayuexin/datasets/Realdex/meshdata', f'{scene_object}.obj')
            # obj_mesh = trimesh.load(mesh_path)

            ######   demo_PADG
            # scene_object = scene_id
            # mesh_path = os.path.join('/inspurfs/group/mayuexin/datasets/PADG_demodataset/mesh', f'{scene_object}.obj')
            # obj_mesh = trimesh.load(mesh_path)

            
            #####   unidex
            # def find_obj_file(scene_object, base_dir):
            #     subdirectories = ['core', 'ddg', 'mujoco', 'sem']
            #     for subdirectory in subdirectories:
            #         mesh_path = os.path.join(base_dir, subdirectory, scene_object, 'coacd', 'decomposed.obj')
            #         if os.path.exists(mesh_path):
            #             return mesh_path
            #     raise FileNotFoundError(f'{scene_object}.obj not found in any subdirectory')
            # scene_object = scene_id
            # mesh_path = find_obj_file(scene_object, '/inspurfs/group/mayuexin/datasets/UniDexGrasp/DFCData/meshdatav3')
            # obj_mesh = trimesh.load(mesh_path)
            # obj_mesh.vertices /= scale
            # #####   unidex
            # scene_object = scene_id
            # mesh_path = os.path.join('/inspurfs/group/mayuexin/datasets/UniDexGrasp/DFCData/obj_scale_urdf', f'{scene_object}.obj')
            # obj_mesh = trimesh.load(mesh_path)

            ###dexgrasp
            # scene_object = scene_id
            # mesh_path = os.path.join('/inspurfs/group/mayuexin/datasets/DexGraspNet/meshdata', scene_object,  'coacd','decomposed.obj')
            # obj_mesh = trimesh.load(mesh_path)
            # obj_mesh.vertices *= scale
            
            ##dexgrasp
            # scene_object = scene_id
            # mesh_path = os.path.join('/inspurfs/group/mayuexin/datasets/DexGraspNet/obj_scale_urdf', f'{scene_object}.obj')
            # obj_mesh = trimesh.load(mesh_path)

            if self.visualize_html:
                for i in range(outputs.shape[0]):
                    # '''
                    self.hand_model.update_kinematics(q=outputs[i:i+1, :])
                    vis_data = [plot_mesh(obj_mesh, color='lightpink')]
                    
                    vis_data += self.hand_model.get_plotly_data(opacity=1.0, color='#8799C6')
                    if i <5:
                        # 保存为 HTML 文件
                        save_path = os.path.join(save_dir, f'{object_name}_sample-{i}.html')
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
                    # '''
                    pbar.update(1)
                # pdb.set_trace()
            res['sample_qpos'][object_name] = np.array(outputs.cpu().detach())
            # rewards_, _ = evaluation_grasp(cfg=cfg, grasps=res, obj_mesh=obj_mesh, device=device, object_name=object_name)
        mean_time = np.mean(cost_time)
        logger.info(f'The inference time is: {mean_time}')
        # rewards_all = evaluation_grasp_all(cfg=cfg, grasps=res, device=device)
        # pickle.dump(res, open(os.path.join(f'/inspurfs/group/mayuexin/zhuyufei/graps_gen/outputs/pkl_file/multidex/{cfg.diffuser.num_inference_steps}', 'res_diffuser.pkl'), 'wb'))

def create_visualizer_val(cfg: DictConfig) -> nn.Module:
    """ Create a visualizer for visual evaluation
    Args:
        cfg: configuration object
        slurm: on slurm platform or not. This field is used to specify the data path
    
    Return:
        A visualizer
    """
    return VISUALIZER.get(cfg.name)(cfg)

