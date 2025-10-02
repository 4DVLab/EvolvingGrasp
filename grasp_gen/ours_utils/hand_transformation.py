import torch
import numpy as np
import transforms3d
import pickle
import json
import os, pdb
from utils.rot6d import normalize_rot6d_torch

T_MIN = [-0.3, -0.3, -0.3]
T_MAX = [0.3, 0.3, 0.3]
NORM_UPPER = 1.0
NORM_LOWER = -1.0

HAND_POSE_MIN = [-1.3174298,  -0.50528544, -0.29243267, -0.5887583,  -1.1498867,  -0.67438334,
                -0.21203011, -0.80237496, -0.8219279,  -0.4905336,  -0.30009413, -0.40315303,
                -0.3069198,  -0.9611522,  -0.28075632, -0.22816268, -0.47040823, -0.76634,
                0.17260288, -0.7636632,  -1.0075126,  -1.7423708, ]
HAND_POSE_MAX = [1.1163847,  1.2857847,  2.4173303,  2.1739862,  1.0859628,  1.3800644,
                2.5540183,  2.266491,   1.0346404,  1.2404677,  2.172077,   2.21667,
                0.8385511,  1.0727106,  1.2466334,  2.0638,     1.8391026,  1.0431478,
                1.6855251,  0.6073609,  0.5251439,  0.24864246,]

def normalize_trans_torch(hand_t):
    t_min = torch.tensor(T_MIN, dtype=hand_t.dtype, device=hand_t.device)
    t_max = torch.tensor(T_MAX, dtype=hand_t.dtype, device=hand_t.device)
    t = torch.div((hand_t - t_min), (t_max - t_min))
    t = t * (NORM_UPPER - NORM_LOWER) - (NORM_UPPER - NORM_LOWER) / 2
    return t

def normalize_rot_torch(hand_r, rot_type='quat'):
    if rot_type == 'mat':
        return normalize_rot6d_torch(hand_r)
    hand_r_min = torch.tensor(eval(f"R_MIN_{rot_type.upper()}"), dtype=hand_r.dtype, device=hand_r.device)
    hand_r_max = torch.tensor(eval(f"R_MAX_{rot_type.upper()}"), dtype=hand_r.dtype, device=hand_r.device)
    r = torch.div((hand_r - hand_r_min), (hand_r_max - hand_r_min))
    r = r * (NORM_UPPER - NORM_LOWER) - (NORM_UPPER - NORM_LOWER) / 2
    return r

def normalize_param_torch(hand_param):
    hand_pose_min = torch.tensor(HAND_POSE_MIN, dtype=hand_param.dtype, device=hand_param.device)
    hand_pose_max = torch.tensor(HAND_POSE_MAX, dtype=hand_param.dtype, device=hand_param.device)
    p = torch.div((hand_param - hand_pose_min), (hand_pose_max - hand_pose_min))
    p = p * (NORM_UPPER - NORM_LOWER) - (NORM_UPPER - NORM_LOWER) / 2
    return p

# -0.5235988, -0.7853982, 
# 0.17453292, 0.61086524, 
# _joint_angle_lower = torch.tensor([-0.43633232, 0., 0., 0., -0.43633232, 0., 0., 0.,
#                                        -0.43633232, 0., 0., 0., 0., -0.43633232, 0., 0., 0., -1.047, 0., -0.2618,
#                                        -0.5237, 0.])
# _joint_angle_upper = torch.tensor([0.43633232, 1.5707964, 1.5707964, 1.5707964, 0.43633232,
#                                        1.5707964, 1.5707964, 1.5707964, 0.43633232, 1.5707964, 1.5707964, 1.5707964,
#                                        0.6981317, 0.43633232, 1.5707964,  1.5707964, 1.5707964, 1.047, 1.309, 0.2618,
#                                        0.5237, 1.])

_joint_angle_lower = torch.tensor([-1.3174298,  -0.50528544, -0.29243267, -0.5887583,  -1.1498867,  -0.67438334,
                -0.21203011, -0.80237496, -0.8219279,  -0.4905336,  -0.30009413, -0.40315303,
                -0.3069198,  -0.9611522,  -0.28075632, -0.22816268, -0.47040823, -0.76634,
                0.17260288, -0.7636632,  -1.0075126,  -1.7423708,])
_joint_angle_upper = torch.tensor([1.1163847,  1.2857847,  2.4173303,  2.1739862,  1.0859628,  1.3800644,
                2.5540183,  2.266491,   1.0346404,  1.2404677,  2.172077,   2.21667,
                0.8385511,  1.0727106,  1.2466334,  2.0638,     1.8391026,  1.0431478,
                1.6855251,  0.6073609,  0.5251439,  0.24864246,])

_global_trans_lower = torch.tensor([-0.3, -0.3, -0.3])
_global_trans_upper = torch.tensor([0.3, 0.3, 0.3])

# _global_trans_lower = torch.tensor([-0.13128923, -0.10665303, -0.45753425])
# _global_trans_upper = torch.tensor([0.12772022, 0.22954416, -0.21764427])

_NORMALIZE_LOWER = -1.
_NORMALIZE_UPPER = 1.

def trans_normalize(global_trans: torch.Tensor, device):
        _global_trans_lower_ = _global_trans_lower.repeat(global_trans.size(0), 1).to(device)
        _global_trans_upper_ = _global_trans_upper.repeat(global_trans.size(0), 1).to(device)
        # _NORMALIZE_UPPER_ = _NORMALIZE_UPPER.to(device)
        # _NORMALIZE_LOWER_ = _NORMALIZE_LOWER.to(device)
        global_trans_norm = torch.div((global_trans - _global_trans_lower_), (_global_trans_upper_ - _global_trans_lower_))
        global_trans_norm = global_trans_norm * (_NORMALIZE_UPPER - _NORMALIZE_LOWER) - (_NORMALIZE_UPPER - _NORMALIZE_LOWER) / 2
        return global_trans_norm

def trans_denormalize(global_trans: torch.Tensor, device):

        global_trans_denorm = global_trans + (self._NORMALIZE_UPPER - self._NORMALIZE_LOWER) / 2
        global_trans_denorm /= (self._NORMALIZE_UPPER - self._NORMALIZE_LOWER)
        global_trans_denorm = global_trans_denorm * (self._global_trans_upper - self._global_trans_lower) + self._global_trans_lower
        return global_trans_denorm

def angle_normalize(joint_angle: torch.Tensor, device):
        _joint_angle_lower_ = _joint_angle_lower.repeat(joint_angle.size(0), 1).to(device)
        _joint_angle_upper_ = _joint_angle_upper.repeat(joint_angle.size(0), 1).to(device)
        # _NORMALIZE_UPPER_ = _NORMALIZE_UPPER.to(device)
        # _NORMALIZE_LOWER_ = _NORMALIZE_LOWER.to(device)
        joint_angle_norm = torch.div((joint_angle - _joint_angle_lower_), (_joint_angle_upper_ - _joint_angle_lower_))
        joint_angle_norm = joint_angle_norm * (_NORMALIZE_UPPER - _NORMALIZE_LOWER) - (_NORMALIZE_UPPER - _NORMALIZE_LOWER) / 2
        return joint_angle_norm

def angle_denormalize(joint_angle: torch.Tensor, device):
        joint_angle_denorm = joint_angle + (self._NORMALIZE_UPPER - self._NORMALIZE_LOWER) / 2
        joint_angle_denorm /= (self._NORMALIZE_UPPER - self._NORMALIZE_LOWER)
        joint_angle_denorm = joint_angle_denorm * (self._joint_angle_upper - self._joint_angle_lower) + self._joint_angle_lower
        return joint_angle_denorm

def load_pkl_file(file_path):
    # Load the pickle file
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data
def load_from_json(input_file):
    with open(input_file, 'r') as f:
        data = json.load(f)
    return data["_train_split"], data["_test_split"], data["_all_split"]
    
def apply_transformations(data, device, rotation_matrixes, global_transl_vec):         # (32, 24), (32, 3, 3)
    new_metadata = []

    for num in range(0, data.size(0)):
        
        # Convert the relevant data to PyTorch tensors
        qpos = data[num].to(device)
        rotation_matrix = rotation_matrixes[num].to(device)

        # Zero out specific angles in qpos
        angle_1 = qpos[0].item()
        angle_2 = qpos[1].item()
        qpos[0] = 0
        qpos[1] = 0

        # Apply rotation transformations
        rotation_matrix_1 = transforms3d.euler.euler2mat(0, angle_1, 0)
        rotation_matrix_1_torch = torch.tensor(rotation_matrix_1, dtype=torch.float32).to(device)

        rotation_matrix_2 = transforms3d.euler.euler2mat(angle_2, 0, 0)
        rotation_matrix_2_torch = torch.tensor(rotation_matrix_2, dtype=torch.float32).to(device)

        combined_rotation_matrix = np.dot(rotation_matrix_1, rotation_matrix_2)
        combined_rotation_matrix_torch = torch.tensor(combined_rotation_matrix, dtype=torch.float32).to(device)

        # Apply translation transformations
        translation_vector = torch.tensor([0, -0.01, 0.2130], dtype=torch.float32).to(device)
        vector = torch.tensor([0, 0, 0.034], dtype=torch.float32).to(device)
        rotated_vector = torch.matmul(rotation_matrix_1_torch, vector.unsqueeze(1)).squeeze(1)
        transl = global_transl_vec.to(device)
        transl = transl - torch.matmul(combined_rotation_matrix_torch, torch.tensor([0, -0.01, 0.2470], dtype=torch.float32).to(device).unsqueeze(1)).squeeze(1) + translation_vector + rotated_vector
        transl = torch.matmul(combined_rotation_matrix_torch.T, transl.T).T

        # Apply rotation matrix transformations
        new_rotation_matrix = torch.matmul(combined_rotation_matrix_torch, rotation_matrix)
        # Save the transformed data in the original format
        qpos = qpos[2:]
        new_metadata.append(qpos)

        # new_metadata.append({
        #     'translations': transl.cpu(),
        #     'rotations': new_rotation_matrix.cpu(),
        #     'joint_positions': qpos.cpu(),
        # })
    
    return torch.stack(new_metadata, dim=0)
'''
def save_new_pt_file(data, new_metadata, file_path):
    # Save the data as a .pt file with torch.save
    new_data = {'info': data['info'], 'metadata': new_metadata}
    torch.save(new_data, file_path)
    print(f"Saved new .pt file to: {file_path}")

# Main script
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
file_path = '/inspurfs/group/mayuexin/datasets/Realdex/realdex_shadowhand_downsample.pt' # '/inspurfs/group/mayuexin/datasets/graspall_datasets/168w.pt'
data = torch.load(os.path.join(file_path))

# Apply transformations
new_metadata = apply_transformations(data, device, )

# Save the new data to a .pt file
new_file_path = '/inspurfs/group/mayuexin/zhuyufei/graps_gen/data/new_realdex_shadowhand_downsample.pt'
save_new_pt_file(data, new_metadata, new_file_path)
'''