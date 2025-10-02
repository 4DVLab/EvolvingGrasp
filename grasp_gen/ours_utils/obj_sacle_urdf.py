# import os
# import pickle
# import trimesh

# # 读取缩放比例
# with open('/inspurfs/group/mayuexin/datasets/DexGraspNet/scales.pkl', 'rb') as f:
#     scales = pickle.load(f)

# # 定义函数生成URDF文件
# def generate_urdf_file(object_name, output_dir):
#     urdf_content = f"""<?xml version="1.0"?>
# <robot name="object">
#   <link name="object">
#     <visual>
#       <origin xyz="0.0 0.0 0.0"/>
#       <geometry>
#         <mesh filename="{object_name}.obj" scale="1.00 1.00 1.00"/>
#       </geometry>
#     </visual>
#     <collision>
#       <origin xyz="0.0 0.0 0.0"/>
#       <geometry>
#         <mesh filename="{object_name}.obj" scale="1.00 1.00 1.00"/>
#       </geometry>
#     </collision>
#   </link>
# </robot>
# """
#     urdf_path = os.path.join(output_dir, f"{object_name}.urdf")
#     with open(urdf_path, 'w') as f:
#         f.write(urdf_content)

# # 查找包含指定 obj 文件的文件夹
# def find_folders_with_obj(directory, obj_filename):
#     folders = []
#     for root, dirs, files in os.walk(directory):
#         if obj_filename in files and os.path.basename(root) == 'coacd':
#             parent_folder = os.path.dirname(root)
#             folders.append(parent_folder)
#     return folders

# # 创建输出目录
# output_dir = '/inspurfs/group/mayuexin/datasets/DexGraspNet/obj_scale_urdf'
# os.makedirs(output_dir, exist_ok=True)

# # 遍历meshdata目录下的所有子目录
# meshdata_dir = '/inspurfs/group/mayuexin/datasets/DexGraspNet/meshdata'
# folders_with_obj = find_folders_with_obj(meshdata_dir, 'decomposed.obj')
# for folder in folders_with_obj:
#     obj_path = os.path.join(folder, 'coacd', 'decomposed.obj')
#     object_name = os.path.basename(folder)
#     if object_name in scales:
#         scale = scales[object_name]
#         # 加载和缩放mesh
#         mesh = trimesh.load(obj_path)
#         mesh.vertices *= scale
#         # 保存缩放后的.obj文件
#         scaled_obj_path = os.path.join(output_dir, f"{object_name}.obj")
#         mesh.export(scaled_obj_path)
#         # 生成URDF文件
#         generate_urdf_file(object_name, output_dir)

# print("缩放和URDF文件生成完毕。")
import os
import pickle
import trimesh

# 读取缩放比例
with open('/inspurfs/group/mayuexin/datasets/UniDexGrasp/DFCData/scales.pkl', 'rb') as f:
    scales = pickle.load(f)

# 定义函数生成URDF文件
def generate_urdf_file(object_name, output_dir):
    urdf_content = f"""<?xml version="1.0"?>
<robot name="object">
  <link name="object">
    <visual>
      <origin xyz="0.0 0.0 0.0"/>
      <geometry>
        <mesh filename="{object_name}.obj" scale="1.00 1.00 1.00"/>
      </geometry>
    </visual>
    <collision>
      <origin xyz="0.0 0.0 0.0"/>
      <geometry>
        <mesh filename="{object_name}.obj" scale="1.00 1.00 1.00"/>
      </geometry>
    </collision>
  </link>
</robot>
"""
    urdf_path = os.path.join(output_dir, f"{object_name}.urdf")
    with open(urdf_path, 'w') as f:
        f.write(urdf_content)

# 查找包含指定 obj 文件的文件夹
def find_folders_with_obj(directory, obj_filename):
    folders = []
    for root, dirs, files in os.walk(directory):
        if obj_filename in files and os.path.basename(root) == 'coacd':
            parent_folder = os.path.dirname(root)
            folders.append(parent_folder)
    return folders

# 创建输出目录
output_dir = '/inspurfs/group/mayuexin/datasets/UniDexGrasp/DFCData/obj_scale_urdf'
os.makedirs(output_dir, exist_ok=True)

# 遍历上一级目录下的四个文件夹
base_dir = '/inspurfs/group/mayuexin/datasets/UniDexGrasp/DFCData/meshdatav3'
subdirectories = [os.path.join(base_dir, sub) for sub in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, sub))]

for subdirectory in subdirectories:
    folders_with_obj = find_folders_with_obj(subdirectory, 'decomposed.obj')
    for folder in folders_with_obj:
        obj_path = os.path.join(folder, 'coacd', 'decomposed.obj')
        object_name = os.path.basename(folder)
        if object_name in scales:
            scale = scales[object_name]
            # 加载和缩放mesh
            mesh = trimesh.load(obj_path)
            mesh.vertices /= scale
            # 保存缩放后的.obj文件
            scaled_obj_path = os.path.join(output_dir, f"{object_name}.obj")
            mesh.export(scaled_obj_path)
            # 生成URDF文件
            generate_urdf_file(object_name, output_dir)

print("缩放和URDF文件生成完毕。")
