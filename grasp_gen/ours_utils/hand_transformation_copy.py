import torch
import numpy as np
import transforms3d
import pickle
import json
import os
import pdb
def load_pkl_file(file_path):
    # Load the pickle file
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data
def load_from_json(input_file):
    with open(input_file, 'r') as f:
        data = json.load(f)
    return data["_train_split"], data["_test_split"], data["_all_split"]
def apply_transformations(data, device):
    new_metadata = []
    
    for object_data in data["metadata"]:
        object_name = object_data['object_name']
        
        # Convert the relevant data to PyTorch tensors
        qpos = object_data['joint_positions'].to(device)
        rotation_matrix = object_data['rotations'].to(device)

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
        transl = object_data['translations'].to(device)
        transl = transl - torch.matmul(combined_rotation_matrix_torch, torch.tensor([0, -0.01, 0.2470], dtype=torch.float32).to(device).unsqueeze(1)).squeeze(1) + translation_vector + rotated_vector
        transl = torch.matmul(combined_rotation_matrix_torch.T, transl.T).T
        pdb.set_trace()
        # Apply rotation matrix transformations
        new_rotation_matrix = torch.matmul(combined_rotation_matrix_torch, rotation_matrix)

        # Save the transformed data in the original format
        new_metadata.append({
            'object_name': object_name,
            'translations': transl.cpu(),
            'rotations': new_rotation_matrix.cpu(),
            'joint_positions': qpos.cpu(),
            'scale': object_data['scale']
        })

    return new_metadata

def save_new_pt_file(data, new_metadata, file_path):
    # Save the data as a .pt file with torch.save
    new_data = {'info': data['info'], 'metadata': new_metadata}
    torch.save(new_data, file_path)
    print(f"Saved new .pt file to: {file_path}")

# Main script
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
file_path = '/inspurfs/group/mayuexin/datasets/MultiDex_UR/shadowhand/need2conbine_data.pt'
data = torch.load(os.path.join(file_path))

# Apply transformations
new_metadata = apply_transformations(data, device)

# Save the new data to a .pt file
new_file_path = '/inspurfs/group/mayuexin/datasets/graspall_datasets/new168w.pt'
# save_new_pt_file(data, new_metadata, new_file_path)

