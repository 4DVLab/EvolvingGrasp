from typing import Any, Tuple, Dict
import os
import json
import pickle
import torch, pdb
import numpy as np
from torch.utils.data import Dataset, DataLoader
from omegaconf import DictConfig, OmegaConf
import trimesh
from datasets.misc import collate_fn_squeeze_pcd_batch_grasp
from datasets.transforms import make_default_transform
from datasets.base import DATASET
from utils.handmodel import get_handmodel
from utils.plotly_utils import plot_mesh
from plotly import graph_objects as go


@DATASET.register()
class MultiDexShadowHandUR(Dataset):
    """ Dataset for pose generation, training with MultiDex Dataset
    """

    _train_split = ["contactdb+alarm_clock", "contactdb+banana", "contactdb+binoculars",
                    "contactdb+cell_phone", "contactdb+cube_large", "contactdb+cube_medium",
                    "contactdb+cube_small", "contactdb+cylinder_large", "contactdb+cylinder_small",
                    "contactdb+elephant", "contactdb+flashlight", "contactdb+hammer",
                    "contactdb+light_bulb", "contactdb+mouse", "contactdb+piggy_bank", "contactdb+ps_controller",
                    "contactdb+pyramid_large", "contactdb+pyramid_medium", "contactdb+pyramid_small",
                    "contactdb+stanford_bunny", "contactdb+stapler", "contactdb+toothpaste", "contactdb+torus_large",
                    "contactdb+torus_medium", "contactdb+torus_small", "contactdb+train",
                    "ycb+bleach_cleanser", "ycb+cracker_box", "ycb+foam_brick", "ycb+gelatin_box", "ycb+hammer",
                    "ycb+lemon", "ycb+master_chef_can", "ycb+mini_soccer_ball", "ycb+mustard_bottle", "ycb+orange",
                    "ycb+peach", "ycb+pitcher_base", "ycb+plum", "ycb+power_drill", "ycb+pudding_box",
                    "ycb+rubiks_cube", "ycb+sponge", "ycb+strawberry", "ycb+sugar_box", "ycb+toy_airplane",
                    "ycb+tuna_fish_can", "ycb+wood_block"]
    #####performace bad: contactdb+camera, cylinder_medium
    # _test_split = ["contactdb+camera", "contactdb+camera", "contactdb+camera"]
    # '''
    # _test_split = ["contactdb+door_knob", "contactdb+door_knob", "contactdb+door_knob"]
    # '''
    # _test_split = ["ycb+tomato_soup_can", "ycb+tomato_soup_can", "ycb+tomato_soup_can", ]#
    # _test_split = ["contactdb+rubber_duck", "contactdb+rubber_duck", "contactdb+rubber_duck"]#
    # _test_split = ["ycb+pear", "ycb+pear","ycb+pear"]# 
    # _test_split = ["ycb+baseball", "ycb+baseball", "ycb+baseball"]#, "ycb+baseball"
    # _test_split = ["contactdb+apple", "contactdb+apple", "contactdb+apple"]#
    # _test_split = ["ycb+potted_meat_can", "ycb+potted_meat_can","ycb+potted_meat_can"]# 
    # _test_split = ["contactdb+cylinder_medium", "contactdb+cylinder_medium", "contactdb+cylinder_medium",]# 
    _test_split = ["contactdb+water_bottle", "contactdb+water_bottle", "contactdb+water_bottle"]# 
    # _test_split = ["contactdb+camera", "contactdb+door_knob", "ycb+tomato_soup_can", "contactdb+rubber_duck", "ycb+pear", "ycb+baseball", "contactdb+apple", 
    # "ycb+potted_meat_can", "contactdb+cylinder_medium", "contactdb+water_bottle"]#
    '''
    _test_split = ["contactdb+camera", "contactdb+camera", "contactdb+camera", "contactdb+camera", "contactdb+camera", "contactdb+camera",
    "contactdb+camera", "contactdb+camera", "contactdb+camera", "contactdb+camera", "contactdb+cylinder_medium", "contactdb+cylinder_medium", "contactdb+cylinder_medium", "contactdb+cylinder_medium", "contactdb+cylinder_medium", "contactdb+cylinder_medium",
    "contactdb+cylinder_medium", "contactdb+cylinder_medium", "contactdb+cylinder_medium", "contactdb+cylinder_medium", "ycb+potted_meat_can", "ycb+potted_meat_can", "ycb+potted_meat_can", "ycb+potted_meat_can", "ycb+potted_meat_can", "ycb+potted_meat_can", "ycb+potted_meat_can",
    "ycb+potted_meat_can", "ycb+potted_meat_can", "ycb+potted_meat_can", "contactdb+apple", "contactdb+apple", "contactdb+apple", "contactdb+apple","contactdb+apple", "contactdb+apple", "contactdb+apple",
     "contactdb+apple", "contactdb+apple", "contactdb+apple", "contactdb+door_knob", "contactdb+door_knob", "contactdb+door_knob", "contactdb+door_knob", "contactdb+door_knob", "contactdb+door_knob", "contactdb+door_knob",
    "contactdb+door_knob", "contactdb+door_knob", "contactdb+door_knob", "contactdb+camera", "contactdb+cylinder_medium", "contactdb+rubber_duck",
                    "contactdb+water_bottle", "ycb+baseball", "ycb+pear", "ycb+tomato_soup_can"]
    '''
    _all_split = ["contactdb+alarm_clock", "contactdb+banana", "contactdb+binoculars",
                  "contactdb+cell_phone", "contactdb+cube_large", "contactdb+cube_medium",
                  "contactdb+cube_small", "contactdb+cylinder_large", "contactdb+cylinder_small",
                  "contactdb+elephant", "contactdb+flashlight", "contactdb+hammer",
                  "contactdb+light_bulb", "contactdb+mouse", "contactdb+piggy_bank", "contactdb+ps_controller",
                  "contactdb+pyramid_large", "contactdb+pyramid_medium", "contactdb+pyramid_small",
                  "contactdb+stanford_bunny", "contactdb+stapler", "contactdb+toothpaste", "contactdb+torus_large",
                  "contactdb+torus_medium", "contactdb+torus_small", "contactdb+train",
                  "ycb+bleach_cleanser", "ycb+cracker_box", "ycb+foam_brick", "ycb+gelatin_box", "ycb+hammer",
                  "ycb+lemon", "ycb+master_chef_can", "ycb+mini_soccer_ball", "ycb+mustard_bottle", "ycb+orange",
                  "ycb+peach", "ycb+pitcher_base", "ycb+plum", "ycb+power_drill", "ycb+pudding_box",
                  "ycb+rubiks_cube", "ycb+sponge", "ycb+strawberry", "ycb+sugar_box", "ycb+toy_airplane",
                  "ycb+tuna_fish_can", "ycb+wood_block", "contactdb+apple", "contactdb+camera", "contactdb+cylinder_medium", "contactdb+rubber_duck",
                  "contactdb+door_knob",  "contactdb+water_bottle", "ycb+baseball", "ycb+pear", "ycb+potted_meat_can",
                  "ycb+tomato_soup_can"]

    _joint_angle_lower = torch.tensor([-0.5235988, -0.7853982, -0.43633232, 0., 0., 0., -0.43633232, 0., 0., 0.,
                                       -0.43633232, 0., 0., 0., 0., -0.43633232, 0., 0., 0., -1.047, 0., -0.2618,
                                       -0.5237, 0.])
    _joint_angle_upper = torch.tensor([0.17453292, 0.61086524, 0.43633232, 1.5707964, 1.5707964, 1.5707964, 0.43633232,
                                       1.5707964, 1.5707964, 1.5707964, 0.43633232, 1.5707964, 1.5707964, 1.5707964,
                                       0.6981317, 0.43633232, 1.5707964,  1.5707964, 1.5707964, 1.047, 1.309, 0.2618,
                                       0.5237, 1.])

    _global_trans_lower = torch.tensor([-0.13128923, -0.10665303, -0.45753425])
    _global_trans_upper = torch.tensor([0.12772022, 0.22954416, -0.21764427])

    _NORMALIZE_LOWER = -1.
    _NORMALIZE_UPPER = 1.

    def __init__(self, cfg: DictConfig, phase: str, slurm: bool, case_only: bool=False, **kwargs: Dict) -> None:
        super(MultiDexShadowHandUR, self).__init__()
        self.phase = phase
        self.slurm = slurm
        if self.phase == 'train':
            self.split = self._train_split
        elif self.phase == 'test':
            self.split = self._test_split
        elif self.phase == 'all':
            self.split = self._all_split
        else:
            raise Exception('Unsupported phase.')
        self.datasetname = 'MultiDexShadowHandUR'
        self.device = cfg.device
        self.is_downsample = cfg.is_downsample
        self.modeling_keys = cfg.modeling_keys

        self.DPO_set_dir = cfg.DPO_set_dir
        self.train_DPO = cfg.train_DPO

        self.num_points = cfg.num_points
        self.use_color = cfg.use_color
        self.use_normal = cfg.use_normal
        self.normalize_x = cfg.normalize_x
        self.normalize_x_trans = cfg.normalize_x_trans
        self.obj_dim = int(3 + 3 * self.use_color + 3 * self.use_normal)
        self.transform = make_default_transform(cfg, phase)
        self.use_llm=cfg.use_llm
        ## resource folders
        self.asset_dir = cfg.asset_dir_slrum if self.slurm else cfg.asset_dir
        self.data_dir = os.path.join(self.asset_dir, 'shadowhand')
        self.scene_path = os.path.join(self.asset_dir, 'object_pcds_nors.pkl')
        self._joint_angle_lower = self._joint_angle_lower.cpu()
        self._joint_angle_upper = self._joint_angle_upper.cpu()
        self._global_trans_lower = self._global_trans_lower.cpu()
        self._global_trans_upper = self._global_trans_upper.cpu()
        self.use_all_pra = cfg.use_all_pra
        ## load data
        self._pre_load_data(case_only)

    def _pre_load_data(self, case_only: bool) -> None:
        """ Load dataset

        Args:
            case_only: only load single case for testing, if ture, the dataset will be smaller.
                        This is useful in after-training visual evaluation.
        """
        self.frames = []
        self.frames_neg = []
        self.scene_pcds = {}
        # grasp_dataset = torch.load(os.path.join(self.data_dir, 'shadowhand_downsample.pt' if self.is_downsample else 'shadowhand.pt'))
        if self.train_DPO:
            grasp_dataset = torch.load(os.path.join(self.DPO_set_dir, 'multidex_combined_final_thesame.pt'))
            # grasp_dataset = torch.load(os.path.join(self.DPO_set_dir, 'multidex_positive_final.pt'))
            # grasp_dataset_neg = torch.load(os.path.join(self.DPO_set_dir, 'multidex_negative_final.pt'))
        else:
            grasp_dataset = torch.load(os.path.join(self.data_dir, 'filter_0.01_shadowhand_downsample.pt' if self.is_downsample else 'shadowhand_downsample.pt'))
        self.scene_pcds = pickle.load(open(self.scene_path, 'rb'))
        # pdb.set_trace()
        if self.use_llm:
            # Getting descriptions from LLM
            scene_text_file = "/inspurfs/group/mayuexin/datasets/dex_hand_preprocessed/multidex_gpt4o_mini.json"
            # scene_text_file = "multidex_gpt4o_mini.json"
            with open(scene_text_file, "r") as jsonfile:
                self.scene_text = json.load(jsonfile)
            # pre-process for tokenizer
            for k, text in self.scene_text.items():
                txtclips = text.split("\n")
                self.scene_text[k] = txtclips[:]
                # if len(txtclips) <= 3: 
                #     txtclips = text.split(".")
                #     self.scene_text[k] = txtclips
                # else:
                #     self.scene_text[k] = txtclips[2:-2]
        self.dataset_info = grasp_dataset['info']
        # pre-process the dataset info
        for obj in grasp_dataset['info']['num_per_object'].keys():
            if obj not in self.split:
                self.dataset_info['num_per_object'][obj] = 0
        # for obj in self.scene_pcds.keys():
        #     self.scene_pcds[obj] = torch.tensor(self.scene_pcds[obj], device=self.device)
        num_step = 0
        # pdb.set_trace()
        if self.train_DPO:
            for mdata in grasp_dataset['metadata']:
                joint_angle = mdata['joint_positions'].clone().detach()
                global_trans = mdata['translations'].clone().detach()
                hand_rot_mat = mdata['rotations'].numpy()
                # pdb.set_trace()
                if self.use_all_pra:
                    global_trans = torch.matmul(torch.from_numpy(hand_rot_mat), global_trans)
                    #####visualize
                    # outputs_3d_rot = torch.tensor(hand_rot_mat.T[:2].reshape([6]))
                    # mdata_qpos = torch.cat([global_trans, outputs_3d_rot, joint_angle], dim=0).requires_grad_(True)
                    # pdb.set_trace()
                    # self.visualize_(mdata_qpos.unsqueeze(0).to(self.device), mdata['object_name'])
                    ######
                if self.normalize_x:
                    joint_angle = self.angle_normalize(joint_angle)
                if self.normalize_x_trans:
                    global_trans = self.trans_normalize(global_trans)
                
                if self.use_all_pra:
                    outputs_3d_rot = torch.tensor(hand_rot_mat.T[:2].reshape([6]))
                    # if self.normalize_x_rot:
                    #     outputs_3d_rot = normalize_rot6d_torch(outputs_3d_rot)
                    mdata_qpos = torch.cat([global_trans, outputs_3d_rot, joint_angle], dim=0).requires_grad_(True)
                    hand_rot_mat = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
                else:
                    mdata_qpos = torch.cat([global_trans, joint_angle], dim=0).requires_grad_(True)

                # mdata_qpos = torch.cat([global_trans, joint_angle], dim=0).requires_grad_(True)

                if mdata['object_name'] in self.split:
                    self.frames.append({'robot_name': 'shadowhand',
                                        'object_name': mdata['object_name'],
                                        'object_rot_mat': hand_rot_mat.T,
                                        'qpos': mdata_qpos,
                                        'label': mdata['label']})
                '''
                for mdata1 in grasp_dataset_neg['metadata']:
                    if mdata1['object_name'] != mdata['object_name']:
                        continue

                    # mdata_qpos = mdata[0].cpu()
                    hand_rot_mat = mdata['rotations'].numpy()
                    joint_angle = mdata['joint_positions'].clone().detach()
                    global_trans = mdata['translations'].clone().detach()
                    # pdb.set_trace()
                    if self.use_all_pra:
                        global_trans = torch.matmul(torch.from_numpy(hand_rot_mat), global_trans)
                        #####visualize
                        # outputs_3d_rot = torch.tensor(hand_rot_mat.T[:2].reshape([6]))
                        # mdata_qpos = torch.cat([global_trans, outputs_3d_rot, joint_angle], dim=0).requires_grad_(True)
                        # pdb.set_trace()
                        # self.visualize_(mdata_qpos.unsqueeze(0).to(self.device), mdata['object_name'])
                        ######
                    if self.normalize_x:
                        joint_angle = self.angle_normalize(joint_angle)
                    if self.normalize_x_trans:
                        global_trans = self.trans_normalize(global_trans)
                    
                    if self.use_all_pra:
                        outputs_3d_rot = torch.tensor(hand_rot_mat.T[:2].reshape([6]))
                        # if self.normalize_x_rot:
                        #     outputs_3d_rot = normalize_rot6d_torch(outputs_3d_rot)
                        mdata_qpos = torch.cat([global_trans, outputs_3d_rot, joint_angle], dim=0).requires_grad_(True)
                        hand_rot_mat = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
                    else:
                        mdata_qpos = torch.cat([global_trans, joint_angle], dim=0).requires_grad_(True)

                    # mdata_qpos = torch.cat([global_trans, joint_angle], dim=0).requires_grad_(True)
                    # pdb.set_trace()
                    if mdata['object_name'] in self.split:
                        self.frames.append({'robot_name': 'shadowhand',
                                            'object_name': mdata['object_name'],
                                            'object_rot_mat': hand_rot_mat.T,
                                            'qpos': mdata_qpos,
                                            'label': -1})
                '''
        else:
            for mdata in grasp_dataset['metadata']:
                num_step = num_step + 1
                # if num_step != 12000:
                #     continue
                # pdb.set_trace()
                mdata_qpos = mdata[0].cpu()
                joint_angle = mdata_qpos.clone().detach()[9:] + 0.5 * torch.rand(24)
                global_trans = mdata_qpos.clone().detach()[:3] + 0.1 * torch.rand(3)
                hand_rot_mat = mdata[1].clone().detach().numpy().T
                # pdb.set_trace()
                if self.use_all_pra:
                    global_trans = torch.matmul(torch.from_numpy(hand_rot_mat), global_trans)
                    #####visualize
                    # outputs_3d_rot = torch.tensor(hand_rot_mat.T[:2].reshape([6]))
                    # mdata_qpos = torch.cat([global_trans, outputs_3d_rot, joint_angle], dim=0).requires_grad_(True)
                    # pdb.set_trace()
                    # self.visualize_(mdata_qpos.unsqueeze(0).to(self.device), mdata[2])
                    ######
                if self.normalize_x:
                    joint_angle = self.angle_normalize(joint_angle)
                if self.normalize_x_trans:
                    global_trans = self.trans_normalize(global_trans)
                
                if self.use_all_pra:
                    outputs_3d_rot = torch.tensor(hand_rot_mat.T[:2].reshape([6]))
                    # if self.normalize_x_rot:
                    #     outputs_3d_rot = normalize_rot6d_torch(outputs_3d_rot)
                    mdata_qpos = torch.cat([global_trans, outputs_3d_rot, joint_angle], dim=0).requires_grad_(True)
                    hand_rot_mat = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
                else:
                    mdata_qpos = torch.cat([global_trans, joint_angle], dim=0).requires_grad_(True)

                # mdata_qpos = torch.cat([global_trans, joint_angle], dim=0).requires_grad_(True)

                if mdata[2] in self.split:
                    self.frames.append({'robot_name': 'shadowhand',
                                        'object_name': mdata[2],
                                        'object_rot_mat': hand_rot_mat.T,
                                        'qpos': mdata_qpos})
    
    def visualize_(self, mdata_qpos, scene_id):
        scene_dataset, scene_object = scene_id.split('+')
        mesh_path = os.path.join('assets/object', scene_dataset, scene_object, f'{scene_object}.stl')
        obj_mesh = trimesh.load(mesh_path)
        hand_model = get_handmodel(batch_size=1, device=self.device)
        for i in range(mdata_qpos.shape[0]):
            hand_model.update_kinematics(q=mdata_qpos[i:i+1, :])
            vis_data = [plot_mesh(obj_mesh, color='lightpink')]
               
            vis_data += hand_model.get_plotly_data(opacity=1.0, color='#8799C6')
            if i <100:
                # 保存为 HTML 文件
                save_path = os.path.join('/inspurfs/group/mayuexin/zhuyufei/graps_gen/outputs/temp', f'{scene_object}_sample-{i}.html')
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

    def trans_normalize(self, global_trans: torch.Tensor):
        global_trans_norm = torch.div((global_trans - self._global_trans_lower), (self._global_trans_upper - self._global_trans_lower))
        global_trans_norm = global_trans_norm * (self._NORMALIZE_UPPER - self._NORMALIZE_LOWER) - (self._NORMALIZE_UPPER - self._NORMALIZE_LOWER) / 2
        return global_trans_norm

    def trans_denormalize(self, global_trans: torch.Tensor):
        global_trans_denorm = global_trans + (self._NORMALIZE_UPPER - self._NORMALIZE_LOWER) / 2
        global_trans_denorm /= (self._NORMALIZE_UPPER - self._NORMALIZE_LOWER)
        global_trans_denorm = global_trans_denorm * (self._global_trans_upper - self._global_trans_lower) + self._global_trans_lower
        return global_trans_denorm

    def angle_normalize(self, joint_angle: torch.Tensor):
        joint_angle_norm = torch.div((joint_angle - self._joint_angle_lower), (self._joint_angle_upper - self._joint_angle_lower))
        joint_angle_norm = joint_angle_norm * (self._NORMALIZE_UPPER - self._NORMALIZE_LOWER) - (self._NORMALIZE_UPPER - self._NORMALIZE_LOWER) / 2
        return joint_angle_norm

    def angle_denormalize(self, joint_angle: torch.Tensor):
        joint_angle_denorm = joint_angle + (self._NORMALIZE_UPPER - self._NORMALIZE_LOWER) / 2
        joint_angle_denorm /= (self._NORMALIZE_UPPER - self._NORMALIZE_LOWER)
        joint_angle_denorm = joint_angle_denorm * (self._joint_angle_upper - self._joint_angle_lower) + self._joint_angle_lower
        return joint_angle_denorm

    def __len__(self):
        return len(self.frames)
    
    def __getitem__(self, index: Any) -> Tuple:
        # pdb.set_trace()
        frame = self.frames[index]

        ## load data, containing scene point cloud and point pose
        scene_id = frame['object_name']
        scene_rot_mat = frame['object_rot_mat']
        scene_pc = self.scene_pcds[scene_id]
        nor = np.einsum('mn, kn->km', scene_rot_mat, scene_pc[:,3:6])
        scene_pc = np.einsum('mn, kn->km', scene_rot_mat, scene_pc[:,:3])
        cam_tran = None

        ## randomly resample points
        if self.phase != 'train':
            np.random.seed(0) # resample point cloud with a fixed random seed
        # np.random.shuffle(scene_pc)
        # scene_pc = scene_pc[:self.num_points]
        resample_indices = np.random.permutation(len(scene_pc))
        scene_pc = scene_pc[resample_indices[:self.num_points]]
        nor = nor[resample_indices[:self.num_points]]
        ## format point cloud xyz and feature
        xyz = scene_pc[:, 0:3]
        nor = nor[:, 0:3]
        if self.use_color:
            color = scene_pc[:, 3:6] / 255.
            feat = np.concatenate([color], axis=-1)

        ## format smplx parameters
        grasp_qpos = (
            frame['qpos']
        )
        if self.train_DPO:
            label = frame['label']
            data = {
                'x': grasp_qpos,
                'pos': xyz,
                'scene_rot_mat': scene_rot_mat,
                'cam_tran': cam_tran, 
                'scene_id': scene_id,
                'normal': nor,
                'label': label,
                # 'text': self.scene_text[scene_id],
            }
        else:
            data = {
                'x': grasp_qpos,
                'pos': xyz,
                'scene_rot_mat': scene_rot_mat,
                'cam_tran': cam_tran, 
                'scene_id': scene_id,
                'normal': nor,
                # 'text': self.scene_text[scene_id],
            }
        '''
        if self.train_DPO:
            frame_neg = self.frames_neg[index]
            ## load data, containing scene point cloud and point pose
            # scene_id = frame_neg['object_name']
            # scene_rot_mat = frame_neg['object_rot_mat']
            # scene_pc_neg = self.scene_pcds[scene_id]
            # nor_neg = np.einsum('mn, kn->km', scene_rot_mat, scene_pc_neg[:,3:6])
            # scene_pc_neg = np.einsum('mn, kn->km', scene_rot_mat, scene_pc_neg[:,:3])
            cam_tran = None

            ## randomly resample points
            if self.phase != 'train':
                np.random.seed(0) # resample point cloud with a fixed random seed
            # np.random.shuffle(scene_pc)
            # scene_pc = scene_pc[:self.num_points]
            # resample_indices_neg = np.random.permutation(len(scene_pc_neg))
            # scene_pc_neg = scene_pc_neg[resample_indices[:self.num_points]]
            # nor_neg = nor_neg[resample_indices[:self.num_points]]
            ## format point cloud xyz and feature
            # xyz_neg = scene_pc_neg[:, 0:3]
            # nor_neg = nor_neg[:, 0:3]
            # if self.use_color:
            #     color_neg = scene_pc_neg[:, 3:6] / 255.
            #     feat_neg = np.concatenate([color_neg], axis=-1)

            ## format smplx parameters
            grasp_qpos_neg = (
                frame_neg['qpos']
            )
            data = {
                'x': grasp_qpos,
                'x_neg': grasp_qpos_neg,
                'pos': xyz,
                # 'pos_neg': xyz_neg,
                'scene_rot_mat': scene_rot_mat,
                'cam_tran': cam_tran, 
                'scene_id': scene_id,
                'normal': nor,
                # 'normal_neg': nor_neg,
                # 'text': self.scene_text[scene_id],
            }
        else:
            data = {
                'x': grasp_qpos,
                'pos': xyz,
                'scene_rot_mat': scene_rot_mat,
                'cam_tran': cam_tran, 
                'scene_id': scene_id,
                'normal': nor,
                # 'text': self.scene_text[scene_id],
            }
        '''
        if self.use_llm:
           data['text'] = self.scene_text[scene_id]

        if self.use_normal:
            normal = nor
            feat = np.concatenate([normal], axis=-1)
            data['feat'] = feat
            # if self.train_DPO:
            #     data['feat_neg'] = feat_neg
        if self.transform is not None:
            data = self.transform(data, modeling_keys=self.modeling_keys)

        return data

    def get_dataloader(self, **kwargs):
        return DataLoader(self, **kwargs)


if __name__ == '__main__':
    config_path = "../configs/task/grasp_gen.yaml"
    cfg = OmegaConf.load(config_path)
    dataloader = MultiDexShadowHandUR(cfg.dataset, 'train', False).get_dataloader(batch_size=4,
                                                                                  collate_fn=collate_fn_squeeze_pcd_batch_grasp,
                                                                                  num_workers=0,
                                                                                  pin_memory=True,
                                                                                  shuffle=True,)

    device = 'cuda'
    for it, data in enumerate(dataloader):
        for key in data:
            if torch.is_tensor(data[key]):
                data[key] = data[key].to(device)
        print()