import torch

def load_pt_file(file_path):
    data = torch.load(file_path)
    return data

# 示例：使用该函数读取文件
# file_path = '/inspurfs/group/mayuexin/zym/diffusion+hand/Scene-Diffuser/MultiDex_UR/shadowhand/shadowhand_downsample.pt'
# file_path = '/inspurfs/group/mayuexin/zym/diffusion+hand/Scene-Diffuser/MultiDex_UR/shadowhand/shadowhand_downsample.pt'
file_path = '/inspurfs/group/mayuexin/datasets/UniDexGrasp/DFCData/unidexgrasp_shadowhand_downsample.pt'
# file_path = '/inspurfs/group/mayuexin/datasets/Realdex/realdex_shadowhand_downsample.pt'
data = load_pt_file(file_path)

# 打印数据
print(list(data.keys()))
print(data['info'])
print(data['metadata'][:20])


# import torch
# import numpy as np
# import transforms3d

# def load_pt_file(file_path):
#     data = torch.load(file_path)
#     return data

# def apply_transformations(data, device):
#     new_metadata = []

#     for item in data['metadata']:
#         # 获取 qpos 和旋转矩阵
#         qpos = item[0].clone()
#         rotation_matrix = item[1].clone()

#         # 设置 qpos[0, 9] 和 qpos[0, 10] 为 0
#         angle_1 = qpos[9].item()
#         angle_2 = qpos[10].item()
#         qpos[9] = 0
#         qpos[10] = 0

#         # 绕 y 轴旋转
#         rotation_matrix_1 = transforms3d.euler.euler2mat(0, angle_1, 0)
#         rotation_matrix_1_torch = torch.tensor(rotation_matrix_1, dtype=torch.float32).to(device)

#         # 绕 x 轴旋转
#         rotation_matrix_2 = transforms3d.euler.euler2mat(angle_2, 0, 0)
#         rotation_matrix_2_torch = torch.tensor(rotation_matrix_2, dtype=torch.float32).to(device)

#         # 矩阵相乘得到组合的旋转矩阵
#         combined_rotation_matrix = np.dot(rotation_matrix_1, rotation_matrix_2)
#         combined_rotation_matrix_torch = torch.tensor(combined_rotation_matrix, dtype=torch.float32).to(device)

#         # 应用变换到 qpos 的前三维度
#         translation_vector = torch.tensor([0, -0.01, 0.2130], dtype=torch.float32).to(device)
#         vector = torch.tensor([0, 0, 0.034], dtype=torch.float32).to(device)
#         rotated_vector = torch.matmul(rotation_matrix_1_torch, vector.unsqueeze(1)).squeeze(1)
#         transl = qpos[:3].unsqueeze(0).to(device)
#         transl = transl - torch.matmul(combined_rotation_matrix_torch, torch.tensor([0, -0.01, 0.2470], dtype=torch.float32).to(device).unsqueeze(1)).squeeze(1) + translation_vector + rotated_vector
#         transl = torch.matmul(combined_rotation_matrix_torch.T, transl.T).T
#         qpos[:3] = transl.squeeze(0)

#         # 应用变换到旋转矩阵
#         new_rotation_matrix = torch.matmul(combined_rotation_matrix_torch.T, rotation_matrix)

#         # 构建新的 metadata 项
#         new_metadata.append((qpos, new_rotation_matrix, item[2], item[3]))

#     return new_metadata

# def save_new_pt_file(data, new_metadata, file_path):
#     new_data = {'info': data['info'], 'metadata': new_metadata}
#     torch.save(new_data, file_path)
#     print(f"保存新的 pt 文件到: {file_path}")

# # 示例：使用该函数读取文件并保存新的文件
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# file_path = '/inspurfs/group/mayuexin/zym/diffusion+hand/Scene-Diffuser/MultiDex_UR/shadowhand/shadowhand_downsample.pt'
# data = load_pt_file(file_path)

# # 应用变换
# new_metadata = apply_transformations(data, device)

# # 保存为新的 pt 文件
# new_file_path = '/inspurfs/group/mayuexin/datasets/MultiDex_UR/shadowhand/shadowhand_downsample.pt'
# save_new_pt_file(data, new_metadata, new_file_path)
# import torch
# import numpy as np
# import transforms3d
# import pickle

# def load_pkl_file(file_path):
#     # Load the pickle file
#     with open(file_path, 'rb') as f:
#         data = pickle.load(f)
#     return data

# def apply_transformations(data, device):
#     new_metadata = {}
    
#     _all_split = ['cylindersmall', 'cylindermedium', 'toruslarge', 'camera', 'train', 'mug', 'knife', 'binoculars', 'spheresmall', 
#                   'airplane', 'torusmedium', 'rubberduck', 'apple', 'cubesmall', 'wristwatch', 'cylinderlarge', 'flute', 'stamp', 
#                   'scissors', 'bowl', 'pyramidlarge', 'toothbrush', 'cubemedium', 'teapot', 'duck', 'gamecontroller', 'hammer', 
#                   'flashlight', 'waterbottle', 'torussmall', 'headphones', 'mouse', 'cubemiddle', 'stapler', 'elephant', 'piggybank', 
#                   'alarmclock', 'cubelarge', 'cup', 'wineglass', 'lightbulb', 'watch', 'phone', 'eyeglasses', 'spherelarge', 'spheremedium', 
#                   'toothpaste', 'doorknob', 'stanfordbunny', 'hand', 'coffeemug', 'pyramidmedium', 'fryingpan', 'table', 'pyramidsmall', 'banana']    

#     for object_name in _all_split:
#         poses_per_object = data["data"].get(object_name, [])  # assuming data['data'] structure
#         new_metadata[object_name] = []  # Initialize a list to store metadata for each object

#         for item in poses_per_object:
#             # Convert the relevant data to PyTorch tensors
#             qpos = torch.tensor(item['joints'], dtype=torch.float32).to(device)
#             rotation_matrix = torch.tensor(item["R"], dtype=torch.float32).to(device)

#             # Zero out specific angles in qpos
#             angle_1 = qpos[0].item()
#             angle_2 = qpos[1].item()
#             qpos[0] = 0
#             qpos[1] = 0

#             # Apply rotation transformations
#             rotation_matrix_1 = transforms3d.euler.euler2mat(0, angle_1, 0)
#             rotation_matrix_1_torch = torch.tensor(rotation_matrix_1, dtype=torch.float32).to(device)

#             rotation_matrix_2 = transforms3d.euler.euler2mat(angle_2, 0, 0)
#             rotation_matrix_2_torch = torch.tensor(rotation_matrix_2, dtype=torch.float32).to(device)

#             combined_rotation_matrix = np.dot(rotation_matrix_1, rotation_matrix_2)
#             combined_rotation_matrix_torch = torch.tensor(combined_rotation_matrix, dtype=torch.float32).to(device)

#             # Apply translation transformations
#             translation_vector = torch.tensor([0, -0.01, 0.2130], dtype=torch.float32).to(device)
#             vector = torch.tensor([0, 0, 0.034], dtype=torch.float32).to(device)
#             rotated_vector = torch.matmul(rotation_matrix_1_torch, vector.unsqueeze(1)).squeeze(1)
#             transl = torch.tensor(item['rotated_t'], dtype=torch.float32).unsqueeze(0).to(device)
#             transl = transl - torch.matmul(combined_rotation_matrix_torch, torch.tensor([0, -0.01, 0.2470], dtype=torch.float32).to(device).unsqueeze(1)).squeeze(1) + translation_vector + rotated_vector
#             transl = torch.matmul(combined_rotation_matrix_torch.T, transl.T).T

#             # Apply rotation matrix transformations
#             new_rotation_matrix = torch.matmul(combined_rotation_matrix_torch.T, rotation_matrix.T)

#             # Save the transformed data
#             new_metadata[object_name].append({
#                 'R': new_rotation_matrix.cpu().numpy(),
#                 't': item['t'],
#                 'rotated_t': transl.squeeze().cpu().numpy(),
#                 'joints': qpos.cpu().numpy(),
#                 'object_mesh_file': item['object_mesh_file'],
#                 'object_contact': item['object_contact']
#             })

#     return new_metadata

# def save_new_pkl_file(data, new_metadata, file_path):
#     # Save the data as a .pkl file
#     new_data = {'info': data['info'], 'data': new_metadata}
#     with open(file_path, 'wb') as f:
#         pickle.dump(new_data, f)
#     print(f"Saved new .pkl file to: {file_path}")

# # Main script
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# file_path = '/inspurfs/group/mayuexin/datasets/DexGRAB/dexgrab.pkl'
# data = load_pkl_file(file_path)

# # Apply transformations
# new_metadata = apply_transformations(data, device)

# # Save the new data to a .pkl file
# new_file_path = '/inspurfs/group/mayuexin/datasets/DexGRAB/newdexgrab.pkl'
# save_new_pkl_file(data, new_metadata, new_file_path)

