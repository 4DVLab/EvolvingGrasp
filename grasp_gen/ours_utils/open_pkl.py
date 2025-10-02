import pickle
import numpy as np
import os
import yaml 
import torch 
import trimesh
from pytorch3d.transforms import quaternion_to_matrix

def load_pkl_file(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def save_array_as_obj(array, obj_filename):
    with open(obj_filename, 'w') as f:
        for vertex in array:
            f.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")



# 示例：使用该函数读取文件
file_path = '/inspurfs/group/mayuexin/datasets/DexGRAB/dexgrab.pkl'
data = load_pkl_file(file_path)

# 打印数据键值
print(data.keys())
print(data['info'])
print(data['data']['cubemiddle'][:5])
# 假设要保存 'ycb+rubiks_cube' 的数组
array_key = 'bottle-1a7ba1f4c892e2da30711cdbdbc73924'
if array_key in data:
    array = np.array(data[array_key])
    print(array)
    obj_filename = f"{array_key}.obj"
    save_array_as_obj(array, obj_filename)
    print(f"Array for {array_key} has been saved as {obj_filename}")
else:
    print(f"Key {array_key} not found in the data.")

# 示例：使用该函数读取文件
# file_path = '/inspurfs/group/mayuexin/datasets/dexycb/pre-processed/20201002_110253_shadow_pose.pkl'
# with open(file_path, 'rb') as f:
#     data = pickle.load(f)
# # 打印数据键值
# print(data.keys())
# print(data)

# with open("/inspurfs/group/mayuexin/datasets/dexycb/raw/20201002-subject-08/20201002_110253/meta.yml", 'r') as f:
#     meta = yaml.safe_load(f)
# print(meta)
# grasp_ind = meta['ycb_grasp_ind']
# obj_id = meta['ycb_ids'][grasp_ind]
# object_mesh_file = os.path.join("/inspurfs/group/mayuexin/datasets/dexycb/raw/models/009_gelatin_box/textured_simple.obj")
# print("object mesh file: ", object_mesh_file)
# obj_trimesh = trimesh.load(object_mesh_file)

# obj_pose = data["poses"][0]["obj_pose"][grasp_ind]
# print(obj_pose)
# obj_t = torch.from_numpy(obj_pose.p[None,:]).to(torch.float32).cuda() # 1x3
# obj_R = quaternion_to_matrix(torch.from_numpy(obj_pose.q[None,:])).to(torch.float32).cuda()
# print(obj_t)
# print(obj_R)
# obj_t_np = obj_t.cpu().numpy().squeeze(0)
# obj_R_np = obj_R.cpu().numpy().squeeze(0)
# obj_transformation_mat = np.eye(4)
# obj_transformation_mat[:3,:3] = obj_R_np
# obj_transformation_mat[:3,3] = obj_t_np
# obj_trimesh.apply_transform(obj_transformation_mat)
# obj_trimesh.export(os.path.join("/inspurfs/group/mayuexin/zym/diffusion+hand/Scene-Diffuser/dexycb_ours", f"{20201002_110253}_{13}_obj_{obj_id}.ply"))
