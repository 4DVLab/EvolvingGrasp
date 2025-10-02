from typing import Any, Tuple, Dict
import os
import pickle
import torch, pdb
import numpy as np
from torch.utils.data import Dataset, DataLoader
from omegaconf import DictConfig, OmegaConf
import transforms3d
from datasets.misc import collate_fn_squeeze_pcd_batch_grasp
from datasets.transforms import make_default_transform
from datasets.base import DATASET
import json
from utils.registry import Registry
from utils.rot6d import rot_to_orthod6d, robust_compute_rotation_matrix_from_ortho6d, random_rot
import trimesh
from utils.handmodel import get_handmodel
from utils.plotly_utils import plot_mesh
from plotly import graph_objects as go
from utils.rot6d import normalize_rot6d_torch

def load_from_json(input_file):
    with open(input_file, 'r') as f:
        data = json.load(f)
    return data["_train_split"], data["_test_split"], data["_all_split"]


@DATASET.register()
class real_dex(Dataset):
    """ Dataset for pose generation, training with RealDex Dataset
    """

    # read json
    input_file = "/inspurfs/group/mayuexin/datasets/Realdex/grasp.json"
    _train_split, _test_split, _all_split = load_from_json(input_file)

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
        super(real_dex, self).__init__()
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
        self.device = cfg.device
        self.is_downsample = cfg.is_downsample
        self.modeling_keys = cfg.modeling_keys
        self.num_points = cfg.num_points
        self.use_color = cfg.use_color
        self.use_normal = cfg.use_normal
        self.normalize_x = cfg.normalize_x
        self.normalize_x_trans = cfg.normalize_x_trans
        self.normalize_x_rot = cfg.normalize_x_rot
        self.obj_dim = int(3 + 3 * self.use_color + 3 * self.use_normal)
        self.transform = make_default_transform(cfg, phase)
        self.use_llm=cfg.use_llm
        self.use_all_pra = cfg.use_all_pra
        self._test_split = ["goji_jar", "small_sprayer",  "yogurt", "body_lotion", "bowling_game_box", "chips",
        "duck_toy", "cosmetics", "sprayer", "box", "cling_wrap", "saltine_cracker", "strawberry_yogurt",
        "guava_blended_juice", "charmander", "dust_cleaning_sprayer", "mildew_remover",
        "bathroom_cleaner", "crisps"]
        # self._test_split = ["cylinder", 'cylinder', "cylinder",]#, 'crisps'
        ## resource folders
        self.asset_dir = cfg.asset_dir_slrum if self.slurm else cfg.asset_dir
        self.data_dir = os.path.join('/inspurfs/group/mayuexin/datasets/Realdex')
        self.scene_path = os.path.join('/inspurfs/group/mayuexin/datasets/Realdex', 'object_pcds_nors.pkl')
        self._joint_angle_lower = self._joint_angle_lower.cpu()
        self._joint_angle_upper = self._joint_angle_upper.cpu()
        self._global_trans_lower = self._global_trans_lower.cpu()
        self._global_trans_upper = self._global_trans_upper.cpu()

        ## load data
        self._pre_load_data(case_only)

    def _pre_load_data(self, case_only: bool) -> None:
        """ Load dataset

        Args:
            case_only: only load single case for testing, if ture, the dataset will be smaller.
                        This is useful in after-training visual evaluation.
        """
        self.frames = []
        self.scene_pcds = {}
        # pdb.set_trace()
        grasp_dataset = torch.load(os.path.join(self.data_dir, '0.01testnewfilter_shadowhand_downsample.pt' if self.is_downsample else 'realdex_shadowhand_downsample.pt'))
        grasp_dataset_filter = torch.load(os.path.join(self.data_dir, '0.01testnewfilter_shadowhand_downsample.pt'))
        # pdb.set_trace()
        self.scene_pcds = pickle.load(open(self.scene_path, 'rb'))
        self.dataset_info = grasp_dataset['info']
        #{'robot_name': 'shadowhand', 'num_total': 16069, 'num_per_object': {'contactdb+rubber_duck': 300, 'contactdb+pyramid_medium': 300, 
        # [{'object_name': 'mujoco-Beyonc_Life_is_But_a_Dream_DVD', 'translations': tensor([-0.0387, -0.1526, -0.0153]), 'rotations': tensor([1.2206, 0.1691, 2.8811]), 'joint_positions': tensor([ 0.0000e+00,  0.0000e+00, -1.5532e-01,  1.2662e-01,  4.5041e-01,
        #  1.5847e-03, -1.4142e-01,  2.3564e-01,  3.4261e-01,  5.5602e-05,
        # -1.1442e-01,  2.4840e-01,  3.5436e-01,  3.9323e-04,  2.2414e-01,
        # -1.5469e-01,  7.4966e-02,  2.5997e-01,  8.2120e-05,  7.3026e-01,
        #  8.7992e-01, -1.6786e-01, -5.0421e-01, -5.6037e-02]), 'scale': 0.08}

        if self.use_llm:
            # Getting descriptions from LLM
            scene_text_file = "/inspurfs/group/mayuexin/datasets/dex_hand_preprocessed/Realdex_gpt4o_mini.json"
            # scene_text_file = "multidex_gpt4o_mini.json"
            with open(scene_text_file, "r") as jsonfile:
                self.scene_text = json.load(jsonfile)
            # pre-process for tokenizer
            for k, text in self.scene_text.items():
                txtclips = text.split("\n")
                self.scene_text[k] = txtclips[:]
                
        # pre-process the dataset info
        for obj in grasp_dataset['info']['num_per_object'].keys():
            if obj not in self.split:
                self.dataset_info['num_per_object'][obj] = 0
        # for obj in self.scene_pcds.keys():
        #     self.scene_pcds[obj] = torch.tensor(self.scene_pcds[obj], device=self.device)
        for mdata in grasp_dataset['metadata']:
            hand_rot_mat = mdata['rotations'].numpy()
            joint_angle = mdata['joint_positions'].clone().detach()
            global_trans = mdata['translations'].clone().detach()
            
            if self.use_all_pra:
                global_trans = torch.matmul(torch.from_numpy(hand_rot_mat), global_trans)
            
            # outputs_3d_rot = torch.tensor(hand_rot_mat.T[:2].reshape([6]))
            # mdata_qpos = torch.cat([global_trans, outputs_3d_rot, joint_angle], dim=0).requires_grad_(True)
            # pdb.set_trace()
            # self.visualize_(mdata_qpos.unsqueeze(0).to(self.device), mdata['object_name'])

            if self.normalize_x:
                joint_angle = self.angle_normalize(joint_angle)
            if self.normalize_x_trans:
                global_trans = self.trans_normalize(global_trans)

            # joint_angle = self.angle_denormalize(joint_angle)
            # global_trans = self.trans_denormalize(global_trans)
            # outputs_3d_rot = torch.tensor(hand_rot_mat.T[:2].reshape([6]))
            # mdata_qpos = torch.cat([global_trans, outputs_3d_rot, joint_angle], dim=0).requires_grad_(True)
            # self.visualize_(mdata_qpos.unsqueeze(0).to(self.device), mdata['object_name'])
            if self.use_all_pra:
                # pdb.set_trace()
                outputs_3d_rot = torch.tensor(hand_rot_mat.T[:2].reshape([6]))
                # if self.normalize_x_rot:
                #     outputs_3d_rot = normalize_rot6d_torch(outputs_3d_rot)
                mdata_qpos = torch.cat([global_trans, outputs_3d_rot, joint_angle], dim=0).requires_grad_(True)
                # self.visualize_(mdata_qpos.unsqueeze(0).to(self.device), mdata['object_name'])
                hand_rot_mat = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
            else:
                mdata_qpos = torch.cat([global_trans, joint_angle], dim=0).requires_grad_(True)
            if mdata['object_name'] in self.split:
                self.frames.append({'robot_name': 'shadowhand',
                                    'object_name': mdata['object_name'],
                                    'object_rot_mat': hand_rot_mat.T,
                                    'qpos': mdata_qpos,
                                    'scale': mdata['scale']})
            # pdb.set_trace()
            # self.visualize_(mdata_qpos.unsqueeze(0).to(self.device), mdata['object_name'])

        # print('Finishing Pre-load in MultiDexShadowHand')

    def visualize_(self, mdata_qpos, scene_object):
        mesh_path = os.path.join('/inspurfs/group/mayuexin/datasets/Realdex/meshdata', f'{scene_object}.obj')
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
        frame = self.frames[index]
        # pdb.set_trace()
        ## load data, containing scene point cloud and point pose
        scale = frame['scale']
        scene_id = frame['object_name']
        scene_rot_mat = frame['object_rot_mat']
        scene_pc = self.scene_pcds[scene_id]
        # 缩放点云
        scene_pc= scene_pc * 1/scale
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
        
        data = {
            'x': grasp_qpos,
            'pos': xyz,
            'scene_rot_mat': scene_rot_mat,
            'cam_tran': cam_tran, 
            'scene_id': scene_id,
            'normal': nor,
            # 'text': self.scene_text[scene_id],
        }

        if self.use_llm:
           data['text'] = self.scene_text[scene_id]
        if self.use_normal:
            normal = nor
            feat = np.concatenate([normal], axis=-1)
            data['feat'] = feat
        if self.transform is not None:
            data = self.transform(data, modeling_keys=self.modeling_keys)

        return data

    def get_dataloader(self, **kwargs):
        return DataLoader(self, **kwargs)


if __name__ == '__main__':
    config_path = "../configs/task/grasp_gen.yaml"
    cfg = OmegaConf.load(config_path)
    dataloader = real_dex(cfg.dataset, 'train', False).get_dataloader(batch_size=4,
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