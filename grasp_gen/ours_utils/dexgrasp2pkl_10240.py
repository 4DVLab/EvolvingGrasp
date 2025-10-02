# # import os
# # import numpy as np
# # import trimesh
# # import pickle

# # def find_folders_with_obj(directory, obj_filename):
# #     folders = []
# #     for root, dirs, files in os.walk(directory):
# #         if obj_filename in files and os.path.basename(root) == 'coacd':
# #             parent_folder = os.path.dirname(root)
# #             folders.append(parent_folder)
# #     return folders

# # def load_point_cloud(obj_file):
# #     mesh = trimesh.load(obj_file)
# #     if not isinstance(mesh, trimesh.Trimesh):
# #         raise ValueError(f"File {obj_file} is not a valid mesh.")
# #     point_cloud = mesh.sample(10240)  # Sample 10240 points from the mesh
# #     return point_cloud

# # def save_point_cloud_data(folders, output_file):
# #     data = {}
# #     for folder in folders:
# #         obj_file = os.path.join(folder, 'coacd', 'decomposed.obj')
# #         point_cloud = load_point_cloud(obj_file)
# #         folder_name = os.path.basename(folder)
# #         data[folder_name] = point_cloud
    
# #     with open(output_file, 'wb') as f:
# #         pickle.dump(data, f)



# # # 使用示例
# # directory = '/inspurfs/group/mayuexin/datasets/UniDexGrasp/DFCData/meshdatav3'
# # output_file = '/inspurfs/group/mayuexin/datasets/UniDexGrasp/DFCData/point_cloud_data_10240.pkl'
# # folders = find_folders_with_obj(directory, 'decomposed.obj')
# # save_point_cloud_data(folders, output_file)

# # print(f"Point cloud data has been saved to {output_file}")



# # import os
# # import numpy as np
# # import trimesh
# # import pickle

# # def find_folders_with_obj(directory, obj_filename):
# #     folders = []
# #     for root, dirs, files in os.walk(directory):
# #         if obj_filename in files and os.path.basename(root) == 'coacd':
# #             parent_folder = os.path.dirname(root)
# #             folders.append(parent_folder)
# #     return folders

# # def load_point_cloud(obj_file):
# #     mesh = trimesh.load(obj_file)
# #     if not isinstance(mesh, trimesh.Trimesh):
# #         raise ValueError(f"File {obj_file} is not a valid mesh.")
# #     point_cloud = mesh.sample(10240)  # Sample 10240 points from the mesh
# #     return point_cloud

# # def save_point_cloud_data(folders, output_file):
# #     data = {}
# #     for folder in folders:
# #         obj_file = os.path.join(folder, 'coacd', 'decomposed.obj')
# #         point_cloud = load_point_cloud(obj_file)
# #         folder_name = os.path.basename(folder)
# #         data[folder_name] = point_cloud
    
# #     with open(output_file, 'wb') as f:
# #         pickle.dump(data, f)

# # def process_directory(directory, output_file):
# #     # 获取directory下的四个文件夹
# #     subdirectories = [os.path.join(directory, sub) for sub in os.listdir(directory) if os.path.isdir(os.path.join(directory, sub))]
    
# #     # 遍历每个子文件夹，查找包含decomposed.obj的文件夹
# #     all_folders = []
# #     for subdirectory in subdirectories:
# #         folders = find_folders_with_obj(subdirectory, 'decomposed.obj')
# #         all_folders.extend(folders)
    
# #     # 保存点云数据
# #     save_point_cloud_data(all_folders, output_file)


# # # 使用示例
# # directory = '/inspurfs/group/mayuexin/datasets/UniDexGrasp/DFCData/meshdatav3'
# # output_file = '/inspurfs/group/mayuexin/datasets/UniDexGrasp/DFCData/point_cloud_data_10240.pkl'
# # process_directory(directory, output_file)

# # print(f"Point cloud data has been saved to {output_file}")
# # import os
# # import numpy as np
# # import trimesh
# # import pickle

# # def find_obj_files(directory):
# #     obj_files = []
# #     for root, dirs, files in os.walk(directory):
# #         for file in files:
# #             if file.endswith('.obj'):
# #                 obj_files.append(os.path.join(root, file))
# #     return obj_files

# # def load_point_cloud(obj_file):
# #     mesh = trimesh.load(obj_file)
# #     if not isinstance(mesh, trimesh.Trimesh):
# #         raise ValueError(f"File {obj_file} is not a valid mesh.")
# #     point_cloud = mesh.sample(10240)  # Sample 10240 points from the mesh
# #     return point_cloud

# # def save_point_cloud_data(obj_files, output_file):
# #     data = {}
# #     for obj_file in obj_files:
# #         point_cloud = load_point_cloud(obj_file)
# #         file_name = os.path.splitext(os.path.basename(obj_file))[0]
# #         data[file_name] = point_cloud
    
# #     with open(output_file, 'wb') as f:
# #         pickle.dump(data, f)

# # def process_directory(directory, output_file):
# #     # 查找所有 .obj 文件
# #     obj_files = find_obj_files(directory)
    
# #     # 保存点云数据
# #     save_point_cloud_data(obj_files, output_file)


# # # 使用示例
# # directory = '/inspurfs/group/mayuexin/datasets/Realdex/meshdata'
# # output_file = '/inspurfs/group/mayuexin/datasets/Realdex/point_cloud_data_10240.pkl'
# # process_directory(directory, output_file)

# # print(f"Point cloud data has been saved to {output_file}")
# # import os
# # import numpy as np
# # import trimesh
# # import pickle

# # def find_obj_files(directory):
# #     obj_files = []
# #     for root, dirs, files in os.walk(directory):
# #         for file in files:
# #             if file.endswith('.obj'):
# #                 obj_files.append(os.path.join(root, file))
# #     return obj_files

# # def load_point_cloud_with_normals(obj_file):
# #     mesh = trimesh.load(obj_file)
# #     if not isinstance(mesh, trimesh.Trimesh):
# #         raise ValueError(f"File {obj_file} is not a valid mesh.")
# #     points, face_indices = mesh.sample(10240, return_index=True)  # Sample 10240 points from the mesh
# #     normals = mesh.face_normals[face_indices]  # Get the normals corresponding to the sampled points
# #     point_cloud_with_normals = np.hstack((points, normals))  # Combine points and normals
# #     return point_cloud_with_normals

# # def save_point_cloud_data_with_normals(obj_files, output_file):
# #     data = {}
# #     for obj_file in obj_files:
# #         point_cloud_with_normals = load_point_cloud_with_normals(obj_file)
# #         file_name = os.path.splitext(os.path.basename(obj_file))[0]
# #         data[file_name] = point_cloud_with_normals
    
# #     with open(output_file, 'wb') as f:
# #         pickle.dump(data, f)

# # def process_directory_with_normals(directory, output_file):
# #     # 查找所有 .obj 文件
# #     obj_files = find_obj_files(directory)
    
# #     # 保存点云数据和法向量
# #     save_point_cloud_data_with_normals(obj_files, output_file)


# # # 使用示例
# # directory = '/inspurfs/group/mayuexin/datasets/Realdex/meshdata'
# # output_file = '/inspurfs/group/mayuexin/datasets/Realdex/object_pcds_nors.pkl'
# # process_directory_with_normals(directory, output_file)

# # print(f"Point cloud data with normals has been saved to {output_file}")

# # import os
# # import numpy as np
# # import trimesh
# # import pickle

# # def find_folders_with_obj(directory, obj_filename):
# #     folders = []
# #     for root, dirs, files in os.walk(directory):
# #         if obj_filename in files and os.path.basename(root) == 'coacd':
# #             parent_folder = os.path.dirname(root)
# #             folders.append(parent_folder)
# #     return folders

# # def load_point_cloud_with_normals(obj_file):
# #     mesh = trimesh.load(obj_file)
# #     if not isinstance(mesh, trimesh.Trimesh):
# #         raise ValueError(f"File {obj_file} is not a valid mesh.")
# #     points, face_indices = mesh.sample(10240, return_index=True)  # Sample 10240 points from the mesh
# #     normals = mesh.face_normals[face_indices]  # Get the normals corresponding to the sampled points
# #     point_cloud_with_normals = np.hstack((points, normals))  # Combine points and normals
# #     return point_cloud_with_normals

# # def save_point_cloud_data_with_normals(folders, output_file):
# #     data = {}
# #     for folder in folders:
# #         obj_file = os.path.join(folder, 'coacd', 'decomposed.obj')
# #         point_cloud_with_normals = load_point_cloud_with_normals(obj_file)
# #         folder_name = os.path.basename(folder)
# #         data[folder_name] = point_cloud_with_normals
    
# #     with open(output_file, 'wb') as f:
# #         pickle.dump(data, f)

# # def process_directory_with_normals(directory, output_file):
# #     # 获取directory下的四个文件夹
# #     subdirectories = [os.path.join(directory, sub) for sub in os.listdir(directory) if os.path.isdir(os.path.join(directory, sub))]
    
# #     # 遍历每个子文件夹，查找包含decomposed.obj的文件夹
# #     all_folders = []
# #     for subdirectory in subdirectories:
# #         folders = find_folders_with_obj(subdirectory, 'decomposed.obj')
# #         all_folders.extend(folders)
    
# #     # 保存点云数据和法向量
# #     save_point_cloud_data_with_normals(all_folders, output_file)


# # # 使用示例
# # directory = '/inspurfs/group/mayuexin/datasets/UniDexGrasp/DFCData/meshdatav3'
# # output_file = '/inspurfs/group/mayuexin/datasets/UniDexGrasp/DFCData/object_pcds_nors.pkl'
# # process_directory_with_normals(directory, output_file)
# # print(f"Point cloud data with normals has been saved to {output_file}")

# # import os
# # import numpy as np
# # import trimesh
# # import pickle

# # def find_folders_with_obj(directory, obj_filename):
# #     folders = []
# #     for root, dirs, files in os.walk(directory):
# #         if obj_filename in files and os.path.basename(root) == 'coacd':
# #             parent_folder = os.path.dirname(root)
# #             folders.append(parent_folder)
# #     return folders

# # def load_point_cloud_with_normals(obj_file):
# #     mesh = trimesh.load(obj_file)
# #     if not isinstance(mesh, trimesh.Trimesh):
# #         raise ValueError(f"File {obj_file} is not a valid mesh.")
# #     points, face_indices = mesh.sample(10240, return_index=True)  # Sample 10240 points from the mesh
# #     normals = mesh.face_normals[face_indices]  # Get the normals corresponding to the sampled points
# #     point_cloud_with_normals = np.hstack((points, normals))  # Combine points and normals
# #     return point_cloud_with_normals

# # def save_point_cloud_data_with_normals(folders, output_file):
# #     data = {}
# #     for folder in folders:
# #         obj_file = os.path.join(folder, 'coacd', 'decomposed.obj')
# #         point_cloud_with_normals = load_point_cloud_with_normals(obj_file)
# #         folder_name = os.path.basename(folder)
# #         data[folder_name] = point_cloud_with_normals
    
# #     with open(output_file, 'wb') as f:
# #         pickle.dump(data, f)

# # # 使用示例
# # directory = '/inspurfs/group/mayuexin/datasets/DexGraspNet/meshdata'
# # output_file = '/inspurfs/group/mayuexin/datasets/DexGraspNet/object_pcds_nors.pkl'
# # folders = find_folders_with_obj(directory, 'decomposed.obj')
# # save_point_cloud_data_with_normals(folders, output_file)

# # print(f"Point cloud data with normals has been saved to {output_file}")

# import pickle
# import numpy as np

# # 读取缩放比例
# with open('/inspurfs/group/mayuexin/datasets/UniDexGrasp/DFCData/scales.pkl', 'rb') as f:
#     scales = pickle.load(f)

# # 读取点云数据
# with open('/inspurfs/group/mayuexin/datasets/UniDexGrasp/DFCData/object_pcds_nors.pkl', 'rb') as f:
#     object_pcds_nors = pickle.load(f)

# # 对每个物体的点云数据进行缩放
# for object_name, data in object_pcds_nors.items():
#     if object_name in scales:
#         scale = scales[object_name]
#         points = data[:, :3]
#         normals = data[:, 3:]
#         scaled_points = points / scale
#         object_pcds_nors[object_name] = np.concatenate((scaled_points, normals), axis=1)

# # 保存缩放后的点云数据
# new_pkl_path = '/inspurfs/group/mayuexin/datasets/UniDexGrasp/DFCData/new_object_pcds_nors.pkl'
# with open(new_pkl_path, 'wb') as f:
#     pickle.dump(object_pcds_nors, f)

# # print("缩放后的点云数据已保存。")
# import pickle
# import numpy as np

# # 读取缩放比例
# with open('/inspurfs/group/mayuexin/datasets/UniDexGrasp/DFCData/scales.pkl', 'rb') as f:
#     scales = pickle.load(f)

# # 读取缩放后的点云数据
# with open('/inspurfs/group/mayuexin/datasets/UniDexGrasp/DFCData/object_pcds_nors.pkl', 'rb') as f:
#     scaled_object_pcds_nors = pickle.load(f)

# # 反向缩放点云数据
# original_object_pcds_nors = {}
# for object_name, scaled_data in scaled_object_pcds_nors.items():
#     if object_name in scales:
#         scale = scales[object_name]
#         scaled_points = scaled_data[:, :3]
#         normals = scaled_data[:, 3:]
#         original_points = scaled_points * scale
#         original_object_pcds_nors[object_name] = np.concatenate((original_points, normals), axis=1)

# # 保存恢复后的点云数据
# original_pkl_path = '/inspurfs/group/mayuexin/datasets/UniDexGrasp/DFCData/recovered_object_pcds_nors.pkl'
# with open(original_pkl_path, 'wb') as f:
#     pickle.dump(original_object_pcds_nors, f)

# print("原始点云数据已恢复并保存。")
import os
import numpy as np
import trimesh
import pickle

def find_off_files(directory):
    """
    查找指定文件夹中的所有 .off 文件
    """
    off_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.off'):
                off_files.append(os.path.join(root, file))
    return off_files

def load_point_cloud_with_normals(off_file):
    """
    加载 .off 网格文件并生成点云数据，包括法向量
    """
    mesh = trimesh.load(off_file)
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"File {off_file} is not a valid mesh.")
    
    # 从网格中采样 10240 个点
    points, face_indices = mesh.sample(10240, return_index=True)
    
    # 获取与采样点对应的法向量
    normals = mesh.face_normals[face_indices]
    
    # 合并点和法向量
    point_cloud_with_normals = np.hstack((points, normals))
    return point_cloud_with_normals

def save_point_cloud_data_with_normals(off_files, output_file):
    """
    保存所有 .off 文件生成的点云数据（带法向量）到一个文件中
    """
    data = {}
    for off_file in off_files:
        # 加载点云和法向量数据
        point_cloud_with_normals = load_point_cloud_with_normals(off_file)
        
        # 使用文件名作为字典中的键
        file_name = os.path.splitext(os.path.basename(off_file))[0]
        data[file_name] = point_cloud_with_normals
    
    # 使用 pickle 序列化保存数据
    with open(output_file, 'wb') as f:
        pickle.dump(data, f)

def process_directory_with_normals(directory, output_file):
    """
    处理目录中的所有 .off 文件，生成带法向量的点云数据并保存
    """
    # 查找所有 .off 文件
    off_files = find_off_files(directory)
    
    # 保存点云数据和法向量
    save_point_cloud_data_with_normals(off_files, output_file)

# 使用示例
directory = '/inspurfs/group/mayuexin/datasets/scaled_PUnetdata/mesh'  # 请将此路径替换为你的文件夹路径
output_file = '/inspurfs/group/mayuexin/datasets/scaled_PUnetdata/object_pcds_nors.pkl'  # 输出文件路径
process_directory_with_normals(directory, output_file)

print(f"Point cloud data with normals has been saved to {output_file}")
