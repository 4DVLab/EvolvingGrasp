# import torch

# def update_info_in_pt_file(file_path):
#     # Load the .pt file
#     data = torch.load(file_path)

#     # Initialize a new info dictionary
#     new_info = {
#         'robot_name': 'shadowhand',
#         'num_total': 0,
#         'num_per_object': {}
#     }

#     # Iterate over metadata to count grasps
#     for item in data['metadata']:
#         object_name = item[2]
#         if object_name in new_info['num_per_object']:
#             new_info['num_per_object'][object_name] += 1
#         else:
#             new_info['num_per_object'][object_name] = 1
#         new_info['num_total'] += 1

#     # Update the info in the data
#     data['info'] = new_info

#     # Save the updated data back to the .pt file
#     torch.save(data, file_path)

#     return new_info

# # Specify the path to your .pt file
# file_path = '/inspurfs/group/mayuexin/zym/diffusion+hand/Scene-Diffuser/MultiDex_UR/shadowhand/testnewfilter_shadowhand_downsample.pt'

# # Update the info and print the result
# updated_info = update_info_in_pt_file(file_path)

# # Print the updated info
# print(f"Updated Robot Name: {updated_info['robot_name']}")
# print(f"Updated Total Grasps: {updated_info['num_total']}")
# print("Updated Grasps per Object:")
# for object_name, count in updated_info['num_per_object'].items():
#     print(f"  {object_name}: {count}")

# import torch
# from collections import defaultdict

# # 假设你的数据文件名为 'data.pt'
# file_path = '/inspurfs/group/mayuexin/datasets/Realdex/0.1testnewfilter_shadowhand_downsample.pt'

# # 加载原始 .pt 文件
# data = torch.load(file_path)

# # 用于统计的字典
# object_counts = defaultdict(int)

# # 遍历 metadata 列表，统计每个物体的出现次数
# for entry in data['metadata']:
#     object_name = entry['object_name']
#     object_counts[object_name] += 1

# # 更新 info 字典中的 num_per_object 和 num_total
# data['info']['num_per_object'] = dict(object_counts)
# data['info']['num_total'] = sum(object_counts.values())

# # 将更新后的数据保存回原始 .pt 文件
# torch.save(data, file_path)

# print("info 文件内容已更新，并已保存到原始 .pt 文件中")
# import torch
# from collections import defaultdict

# # 加载 .pt 文件
# data = torch.load('/inspurfs/group/mayuexin/datasets/DexGraspNet/scaled_files/filter_data_scale_0_11999999084472726.pt')
# print(data["info"])
# # print(data['metadata'])
# # 获取 metadata 信息
# metadata = data.get('metadata', defaultdict(list))

# # 初始化物体数量统计字典
# object_counts = defaultdict(int)

# # 统计每个物体的数量

# for object_name, object_data_list in metadata.items():
#     object_counts[object_name] = len(object_data_list)
 

# # 计算总数
# total_count = sum(object_counts.values())

# # 打印结果
# print(f"物体种类数量: {len(object_counts)}")
# print(f"每个物体的数量: {dict(object_counts)}")
# print(f"总物体数量: {total_count}")



import torch
from collections import defaultdict

# 加载 .pt 文件
data = torch.load('/inspurfs/group/mayuexin/datasets/grasp_anyting/8_22_30merged_data_unique.pt')
print(data["info"])

# 获取 metadata 信息
metadata = data.get('metadata', [])

# 初始化物体数量统计字典
object_counts = defaultdict(int)

# 统计每个物体的数量
if isinstance(metadata, list):
    for object_data in metadata:
        object_name = object_data['object_name']  # 假设object_name在元组的第三个位置
        object_counts[object_name] += 1
else:
    print("metadata 不是一个列表。请检查数据结构。")

# 计算总数
total_count = sum(object_counts.values())

# 打印结果
print(f"物体种类数量: {len(object_counts)}")
print(f"每个物体的数量: {dict(object_counts)}")
print(f"总物体数量: {total_count}")

# 更新 info 字典中的 num_per_object 和 num_total
data['info']['num_per_object'] = dict(object_counts)
data['info']['num_total'] = total_count

# 将更新后的数据保存回原始 .pt 文件
torch.save(data, '/inspurfs/group/mayuexin/datasets/grasp_anyting/8_22_30merged_data_unique.pt')

print("info 文件内容已更新，并已保存到原始 .pt 文件中")
import torch
import json

# 文件路径
file_path = '/inspurfs/group/mayuexin/datasets/grasp_anyting/8_22_30merged_data_unique.pt'
output_json_path = '/inspurfs/group/mayuexin/datasets/grasp_anyting/objects_above_300.json'

# 加载 .pt 文件
data = torch.load(file_path)

# 提取 'num_per_object' 字典
num_per_object = data['info']['num_per_object']

# 筛选出数量超过300的物体名称
objects_above_300 = {obj: count for obj, count in num_per_object.items() if count > 200}

# 将筛选结果保存到 JSON 文件
with open(output_json_path, 'w') as json_file:
    json.dump(objects_above_300, json_file, indent=4)

print(f"筛选结果已保存到 {output_json_path}")