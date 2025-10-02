# import os
# import shutil

# def find_and_copy_files(source_dir, destination_dir):
#     for root, dirs, files in os.walk(source_dir):
#         # 检查当前目录是否包含这两个文件
#         if 'final_data.npy' in files and 'contact_v2.txt' in files:
#             # 获取相对路径并创建目标文件夹
#             relative_path = os.path.relpath(root, source_dir)
#             destination_folder = os.path.join(destination_dir, relative_path)
#             os.makedirs(destination_folder, exist_ok=True)
            
#             # 复制文件
#             shutil.copy(os.path.join(root, 'final_data.npy'), os.path.join(destination_folder, 'final_data.npy'))
#             shutil.copy(os.path.join(root, 'contact_v2.txt'), os.path.join(destination_folder, 'contact_v2.txt'))
#             print(f"Copied to {destination_folder}")

# source_dir = '/storage/group/4dvlab/youzhuo/RealDex/bags/'  # 源目录
# destination_dir = '/inspurfs/group/mayuexin/datasets/Realdex'  # 目标目录

# find_and_copy_files(source_dir, destination_dir)

import os
import shutil

def get_folder_names(directory):
    # 获取目录下的所有文件夹名
    folder_names = [name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name))]
    return set(folder_names)  # 使用集合以便快速查找

def copy_3d_models(source_dir, target_dir, valid_names):
    # 确保目标目录存在
    os.makedirs(target_dir, exist_ok=True)

    for root, dirs, files in os.walk(source_dir):
        for file in files:
            file_name, file_ext = os.path.splitext(file)
            if file_name in valid_names and file_ext in ['.obj', '.stl', '.fbx']:  # 检查文件扩展名是否为3D模型格式
                src_file_path = os.path.join(root, file)
                dst_file_path = os.path.join(target_dir, file)
                shutil.copy(src_file_path, dst_file_path)
                print(f"Copied: {src_file_path} to {dst_file_path}")

# 指定源目录、目标目录和包含文件夹名的目录
source_dir = '/storage/group/4dvlab/youzhuo/RealDex/models'
target_dir = '/inspurfs/group/mayuexin/datasets/Realdex/meshdata'
folder_names_dir = '/inspurfs/group/mayuexin/datasets/Realdex/data'

# 获取有效的文件夹名
valid_names = get_folder_names(folder_names_dir)

# 复制符合条件的3D模型文件
copy_3d_models(source_dir, target_dir, valid_names)

