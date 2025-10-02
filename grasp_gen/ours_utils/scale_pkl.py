import os
import torch
import pickle

# 加载 .pt 文件
file_path = '/inspurfs/group/mayuexin/datasets/DexGraspNet/dexgraspnet_shadowhand_downsample.pt'
data = torch.load(file_path)

# 提取每个物体的scale
object_scales = {}
for item in data['metadata']:
    object_name = item['object_name']
    scale = item['scale']
    if object_name not in object_scales:
        object_scales[object_name] = []
    object_scales[object_name].append(scale)

# 计算每个物体的平均scale
average_scales = {object_name: sum(scales) / len(scales) for object_name, scales in object_scales.items()}

# 保存成一个.pkl文件
output_file = '/inspurfs/group/mayuexin/datasets/DexGraspNet/scales.pkl'
with open(output_file, 'wb') as f:
    pickle.dump(average_scales, f)

print(f"Average scales have been saved to {output_file}")
