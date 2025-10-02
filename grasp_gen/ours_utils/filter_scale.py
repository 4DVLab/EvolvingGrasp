# import torch
# import os
# from collections import defaultdict
# import numpy as np
# # 加载数据
# file_path = '/inspurfs/group/mayuexin/datasets/UniDexGrasp/DFCData/unidexgrasp_shadowhand_downsample.pt'
# data = torch.load(file_path)

# # 创建一个字典，根据 scale 对数据进行分类
# scale_dict = defaultdict(list)
# for entry in data['metadata']:
#     scale = entry['scale']
#     if isinstance(scale, np.ndarray):
#         scale = scale.item()
#     scale_dict[scale].append(entry)

# # 初始化物体数量统计字典
# object_counts = defaultdict(int)

# # 为每个 scale 保存一个独立的文件，并统计每个物体的数量
# output_dir = '/inspurfs/group/mayuexin/datasets/UniDexGrasp/DFCData/scaled_files/'
# os.makedirs(output_dir, exist_ok=True)

# for scale, entries in scale_dict.items():
#     # 初始化 info 字典
#     info = {
#         'num_per_object': defaultdict(int),
#         'num_total': 0
#     }
    
#     # 统计每个物体的数量
#     for entry in entries:
#         object_name = entry['object_name']
#         info['num_per_object'][object_name] += 1
    
#     # 计算总数
#     info['num_total'] = sum(info['num_per_object'].values())
    
#     # 将 entries 存储在 metadata 键下，并添加统计信息到 info 键
#     output_data = {
#         'info': dict(info),  # 将 defaultdict 转换为普通字典
#         'metadata': entries,

#     }
    
#     output_file = os.path.join(output_dir, f'data_scale_{scale}.pt')
#     torch.save(output_data, output_file)  # 保存到对应的文件中
    
#     print(f"Saved {len(entries)} entries with scale {scale} to {output_file}")
#     print(f"Info for scale {scale}: {output_data['info']}")

# print("All scale-specific files have been saved.")

# import os
# import shutil
# import numpy as np
# import pickle

# # 定义原始目录路径
# source_dir = '/inspurfs/group/mayuexin/datasets/UniDexGrasp/DFCData/obj_scale_urdf'
# # 定义缩放比例
# scales = [6.666666507720947, 8.333333969116211, 16.666667938232422, 10, 12.5]
# # 定义目标目录路径
# target_base_dir = '/inspurfs/group/mayuexin/datasets/UniDexGrasp/DFCData/scaled_obj_urdf'

# # 加载 scales.pkl 文件
# scale_pkl_path = '/inspurfs/group/mayuexin/datasets/UniDexGrasp/DFCData/scales.pkl'
# with open(scale_pkl_path, 'rb') as f:
#     object_scales = pickle.load(f)

# # 创建目标目录下的5个子目录
# output_dirs = []
# for scale in scales:
#     output_dir = os.path.join(target_base_dir, f'scale_{scale}')
#     os.makedirs(output_dir, exist_ok=True)
#     output_dirs.append(output_dir)

# # 处理所有的 .obj 文件
# for file_name in os.listdir(source_dir):
#     if file_name.endswith('.obj'):
#         obj_file_path = os.path.join(source_dir, file_name)
        
#         # 获取物体名称（假设物体名称与 .obj 文件名相同，不带扩展名）
#         object_name = file_name.replace('.obj', '')
        
#         # 从 scales.pkl 中获取物体的缩放比例
#         object_scale = object_scales.get(object_name, 1.0)  # 如果物体没有在 scales.pkl 中找到，默认 scale 为 1.0
        
#         # 读取 .obj 文件
#         with open(obj_file_path, 'r') as obj_file:
#             obj_data = obj_file.readlines()
        
#         # 对每个缩放比例进行处理
#         for scale, output_dir in zip(scales, output_dirs):
#             scaled_obj_file_path = os.path.join(output_dir, file_name)
#             with open(scaled_obj_file_path, 'w') as scaled_obj_file:
#                 for line in obj_data:
#                     # 只缩放顶点 (以 'v ' 开头的行)
#                     if line.startswith('v '):
#                         parts = line.split()
#                         vertex = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
#                         scaled_vertex = vertex / scale * object_scale  # 对每个物体的顶点进行scale调整，并除以物体的特定缩放比例
#                         scaled_obj_file.write(f"v {scaled_vertex[0]} {scaled_vertex[1]} {scaled_vertex[2]}\n")
#                     else:
#                         scaled_obj_file.write(line)
#             print(f"Saved scaled .obj file to: {scaled_obj_file_path}")

# # 复制所有 .urdf 文件到每个目标目录
# for file_name in os.listdir(source_dir):
#     if file_name.endswith('.urdf'):
#         urdf_file_path = os.path.join(source_dir, file_name)
        
#         for output_dir in output_dirs:
#             shutil.copy(urdf_file_path, os.path.join(output_dir, file_name))
#             print(f"Copied .urdf file to: {os.path.join(output_dir, file_name)}")

# print("All files have been processed and saved.")
