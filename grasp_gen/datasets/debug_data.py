import torch
from omegaconf import OmegaConf
from datasets.DexGRAB import DexGRAB
from datasets.misc import collate_fn_squeeze_pcd_batch_grasp

import os
import torch
import trimesh
from plotly import graph_objects as go
from typing import Any

from utils.handmodel import get_handmodel
from utils.plotly_utils import plot_mesh
from utils.rot6d import rot_to_orthod6d, robust_compute_rotation_matrix_from_ortho6d, random_rot


config_path = "configs/task/grasp_gen_ur.yaml"
cfg = OmegaConf.load(config_path)
dataloader = DexGRAB(cfg.dataset, 'train', False).get_dataloader(batch_size=8,
                                                                collate_fn=collate_fn_squeeze_pcd_batch_grasp,
                                                                num_workers=0,
                                                                pin_memory=True,
                                                                shuffle=True,)
"""
    data = {
        'x': grasp_qpos,
        'pos': xyz,
        'scene_rot_mat': scene_rot_mat,
        'cam_tran': cam_tran, 
        'scene_id': scene_id,
    }
"""
hand_model = get_handmodel(batch_size=1, device='cpu')
save_dir = '/public/home/jiangqi2022/robotgen/code/Scene-Diffuser/outputs/debug'

for i, sample in enumerate(dataloader):
    if i>5: break
    outputs = sample['x'].to(torch.float64)   # (B, 27)
    i_rot = torch.stack(list(map(torch.from_numpy, sample['scene_rot_mat']))).to(torch.float64)     # (B, 3, 3)
    scene_id = sample['scene_id']    # (B,) ?

    batch_size = outputs.shape[0]
    ## denormalization
    if dataloader.dataset.normalize_x:
        outputs[:, 3:] = dataloader.dataset.angle_denormalize(joint_angle=outputs[:, 3:])
    if dataloader.dataset.normalize_x_trans:
        outputs[:, :3] = dataloader.dataset.trans_denormalize(global_trans=outputs[:, :3])

    id_6d_rot = torch.tensor([1., 0., 0., 0., 1., 0.]).view(1, 6).repeat(batch_size, 1).to(torch.float64)
    outputs_3d_rot = rot_to_orthod6d(torch.bmm(i_rot.transpose(1,2), robust_compute_rotation_matrix_from_ortho6d(id_6d_rot)))
    outputs[:, :3] = torch.bmm(i_rot.transpose(1,2), outputs[:, :3].unsqueeze(-1)).squeeze(-1)
    outputs = torch.cat([outputs[:, :3], outputs_3d_rot, outputs[:, 3:]], dim=-1)

    os.makedirs(os.path.join(save_dir, 'html'), exist_ok=True)
    for i in range(outputs.shape[0]):
        mesh_path = os.path.join(dataloader.dataset.scene_path, scene_id[i] + '.ply')
        obj_mesh = trimesh.load(mesh_path)
        hand_model.update_kinematics(q=outputs[i:i+1, :])
        vis_data = [plot_mesh(obj_mesh, color='lightblue')]
        vis_data += hand_model.get_plotly_data(opacity=1.0, color='pink')
        save_path = os.path.join(save_dir, 'html', f'{scene_id[i]}+sample-{i}.html')
        fig = go.Figure(data=vis_data)
        fig.write_html(save_path)
