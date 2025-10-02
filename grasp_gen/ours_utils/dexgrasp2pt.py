# import os
# import torch
# import numpy as np
# import transforms3d

# data_path = "/inspurfs/group/mayuexin/datasets/DexGraspNet/dexgraspnet"
# output_path = "/inspurfs/group/mayuexin/datasets/DexGraspNet"

# # Joint names
# joint_names = [
#     'robot0:FFJ3', 'robot0:FFJ2', 'robot0:FFJ1', 'robot0:FFJ0',
#     'robot0:MFJ3', 'robot0:MFJ2', 'robot0:MFJ1', 'robot0:MFJ0',
#     'robot0:RFJ3', 'robot0:RFJ2', 'robot0:RFJ1', 'robot0:RFJ0',
#     'robot0:LFJ4', 'robot0:LFJ3', 'robot0:LFJ2', 'robot0:LFJ1', 'robot0:LFJ0',
#     'robot0:THJ4', 'robot0:THJ3', 'robot0:THJ2', 'robot0:THJ1', 'robot0:THJ0'
# ]

# # Initialize info dictionary
# info = {
#     'robot_name': 'shadowhand',
#     'num_total': 0,
#     'num_per_object': {}
# }

# # Function to convert npy to a single pt file and save
# def convert_npy_to_single_pt(data_path, output_path):
#     combined_data = []

#     # Walk through the directory
#     for root, dirs, files in os.walk(data_path):
#         for file in files:
#             if file.endswith('.npy'):
#                 npy_file_path = os.path.join(root, file)
#                 object_name = os.path.splitext(file)[0]  # Use file name without extension as object name

#                 data = np.load(npy_file_path, allow_pickle=True)

#                 for entry in data:
#                     qpos = entry['qpos']
#                     scale = entry['scale']

#                     if object_name not in info['num_per_object']:
#                         info['num_per_object'][object_name] = 0
#                     info['num_per_object'][object_name] += 1
#                     info['num_total'] += 1

#                     # Extract translations and rotations
#                     translations = torch.tensor([qpos['WRJTx'], qpos['WRJTy'], qpos['WRJTz']])
#                     euler_angles = [qpos['WRJRx'], qpos['WRJRy'], qpos['WRJRz']]
#                     rotation_matrix = transforms3d.euler.euler2mat(*euler_angles)
#                     rotations = torch.tensor(rotation_matrix, dtype=torch.float32)
                    
#                     # Transform translations
#                     transformed_translations = torch.matmul(rotations.T, translations) + torch.tensor([0, 0.0100, -0.2470], dtype=torch.float32)

#                     # Extract joint positions
#                     joint_positions = torch.tensor([0.0, 0.0] + [qpos[joint] for joint in joint_names])
#                     joint_positions[-2]=-1*joint_positions[-2]
#                     joint_positions[6]=-1*joint_positions[6]
#                     joint_positions[2]=-1*joint_positions[2]
#                     # Combine all information
#                     combined_entry = {
#                         'object_name': object_name,
#                         'translations': transformed_translations,
#                         'rotations': rotations,
#                         'joint_positions': joint_positions,
#                         'scale': scale
#                     }

#                     combined_data.append(combined_entry)

#     # Save the combined data and info dictionary to a single pt file
#     pt_file_path = os.path.join(output_path, "dexgraspnet_shadowhand_downsample.pt")
#     torch.save({'info': info, 'metadata': combined_data}, pt_file_path)
#     print(f"Saved combined dataset to {pt_file_path}")
# # Execute the conversion
# convert_npy_to_single_pt(data_path, output_path)



import os
import torch
import numpy as np
import transforms3d

data_path = "/inspurfs/group/mayuexin/datasets/UniDexGrasp/DFCData/datasetv4.1"
output_path = "/inspurfs/group/mayuexin/datasets/UniDexGrasp/DFCData"

# Joint names
joint_names = [
    'robot0:FFJ3', 'robot0:FFJ2', 'robot0:FFJ1', 'robot0:FFJ0',
    'robot0:MFJ3', 'robot0:MFJ2', 'robot0:MFJ1', 'robot0:MFJ0',
    'robot0:RFJ3', 'robot0:RFJ2', 'robot0:RFJ1', 'robot0:RFJ0',
    'robot0:LFJ4', 'robot0:LFJ3', 'robot0:LFJ2', 'robot0:LFJ1', 'robot0:LFJ0',
    'robot0:THJ4', 'robot0:THJ3', 'robot0:THJ2', 'robot0:THJ1', 'robot0:THJ0'
]

# Initialize info dictionary
info = {
    'robot_name': 'shadowhand',
    'num_total': 0,
    'num_per_object': {}
}

# Function to convert npy to a single pt file and save
def convert_npy_to_single_pt(data_path, output_path):
    combined_data = []

    # Walk through the directory
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.endswith('.npz'):
                npz_file_path = os.path.join(root, file)
                object_name = os.path.basename(root)

                data = np.load(npz_file_path, allow_pickle=True)
                
                if isinstance(data['qpos'], np.ndarray):
                    qpos = {str(k): v for k, v in data['qpos'].item().items()}
                else:
                    qpos = {str(k): v for k, v in data['qpos'].items()}

                if object_name not in info['num_per_object']:
                    info['num_per_object'][object_name] = 0

                info['num_per_object'][object_name] += 1
                info['num_total'] += 1
                
                scale = data['scale']

                # Extract translations and rotations
                translations = torch.tensor([qpos['WRJTx'], qpos['WRJTy'], qpos['WRJTz']])
                euler_angles = [qpos['WRJRx'], qpos['WRJRy'], qpos['WRJRz']]
                rotation_matrix = transforms3d.euler.euler2mat(*euler_angles)
                rotations = torch.tensor(rotation_matrix, dtype=torch.float32)
                    
                    # Transform translations
                transformed_translations = torch.matmul(rotations.T, translations) + torch.tensor([0, 0.0100, -0.2470], dtype=torch.float32)
                # Extract joint positions
                joint_positions = torch.tensor([0.0, 0.0] + [qpos[joint] for joint in joint_names])
                joint_positions[-2]=-1*joint_positions[-2]
                joint_positions[6]=-1*joint_positions[6]
                joint_positions[2]=-1*joint_positions[2]

                # Combine all information
                combined_entry = {
                    'object_name': object_name,
                    'translations': transformed_translations,
                    'rotations': rotations,
                    'joint_positions': joint_positions,
                    'scale': scale
                }

                combined_data.append(combined_entry)

    # Save the combined data and info dictionary to a single pt file
    pt_file_path = os.path.join(output_path, "unidexgrasp_shadowhand_downsample1.pt")
    torch.save({'info': info, 'metadata': combined_data}, pt_file_path)
    print(f"Saved combined dataset to {pt_file_path}")

# Execute the conversion
convert_npy_to_single_pt(data_path, output_path)

# import os
# import torch
# import numpy as np

# data_path = "/inspurfs/group/mayuexin/datasets/Realdex/data"
# output_path = "/inspurfs/group/mayuexin/datasets/Realdex"
# joint_names = [
#     'robot0:FFJ3', 'robot0:FFJ2', 'robot0:FFJ1', 'robot0:FFJ0',
#     'robot0:MFJ3', 'robot0:MFJ2', 'robot0:MFJ1', 'robot0:MFJ0',
#     'robot0:RFJ3', 'robot0:RFJ2', 'robot0:RFJ1', 'robot0:RFJ0',
#     'robot0:LFJ4', 'robot0:LFJ3', 'robot0:LFJ2', 'robot0:LFJ1', 'robot0:LFJ0',
#     'robot0:THJ4', 'robot0:THJ3', 'robot0:THJ2', 'robot0:THJ1', 'robot0:THJ0'
# ]

# info = {
#     'robot_name': 'shadowhand',
#     'num_total': 0,
#     'num_per_object': {}
# }

# def load_contact_indices(contact_file):
#     with open(contact_file, 'r') as f:
#         indices = [int(line.strip()) for line in f]
#     return indices

# def convert_npy_to_single_pt(data_path, output_path):
#     combined_data = []
#     for root, dirs, files in os.walk(data_path):
#         for file in files:
#             if file == 'final_data.npy':
#                 npy_file_path = os.path.join(root, file)
#                 contact_file_path = os.path.join(root, 'contact_v2.txt')
#                 object_name = os.path.basename(os.path.dirname(root))

#                 data = np.load(npy_file_path, allow_pickle=True).item()
#                 contact_indices = load_contact_indices(contact_file_path)

#                 if object_name not in info['num_per_object']:
#                     info['num_per_object'][object_name] = 0

#                 for idx in contact_indices:
#                     if idx >= len(data['global_transl']) or idx >= len(data['global_orient']) or idx >= len(data['qpos']) or idx >= len(data['object_transl']) or idx >= len(data['object_orient']):
#                         print(f"Skipping index {idx} for {object_name} as it is out of bounds")
#                         continue

#                     translations = torch.tensor(data['global_transl'][idx], dtype=torch.float32)
#                     rotations = torch.tensor(data['global_orient'][idx], dtype=torch.float32)
#                     joint_positions = torch.cat((torch.tensor([0.0, 0.0], dtype=torch.float32), torch.tensor(data['qpos'][idx], dtype=torch.float32)), dim=0)
#                     object_translations = torch.tensor(data['object_transl'][idx], dtype=torch.float32)
#                     object_orientations = torch.tensor(data['object_orient'][idx], dtype=torch.float32)

#                     combined_entry = {
#                         'object_name': object_name,
#                         'translations': rotations.T @ (translations - object_translations),
#                         'rotations': object_orientations.T @ rotations,
#                         'joint_positions': joint_positions,
#                         'scale': 1.0  # Assuming a scale of 1.0, update if needed
#                     }

#                     combined_data.append(combined_entry)
#                     info['num_per_object'][object_name] += 1
#                     info['num_total'] += 1

#     pt_file_path = os.path.join(output_path, "realdex_shadowhand_downsample.pt")
#     torch.save({'info': info, 'metadata': combined_data}, pt_file_path)
#     print(f"Saved combined dataset to {pt_file_path}")

# convert_npy_to_single_pt(data_path, output_path)





