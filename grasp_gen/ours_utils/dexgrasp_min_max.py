# import os
# import torch
# import numpy as np
# import transforms3d

# # PT文件路径
# pt_file_path = '/inspurfs/group/mayuexin/datasets/DexGraspNet/dexgraspnet_shadowhand_downsample.pt'

# # Joint names
# joint_names = [
#     'robot0:FFJ3', 'robot0:FFJ2', 'robot0:FFJ1', 'robot0:FFJ0',
#     'robot0:MFJ3', 'robot0:MFJ2', 'robot0:MFJ1', 'robot0:MFJ0',
#     'robot0:RFJ3', 'robot0:RFJ2', 'robot0:RFJ1', 'robot0:RFJ0',
#     'robot0:LFJ4', 'robot0:LFJ3', 'robot0:LFJ2', 'robot0:LFJ1', 'robot0:LFJ0',
#     'robot0:THJ4', 'robot0:THJ3', 'robot0:THJ2', 'robot0:THJ1', 'robot0:THJ0'
# ]

# # 初始化min和max张量
# joint_angle_min = torch.full((2+len(joint_names),), float('inf'))
# joint_angle_max = torch.full((2+len(joint_names),), float('-inf'))
# trans_min = torch.full((3,), float('inf'))
# trans_max = torch.full((3,), float('-inf'))
# rot_min = torch.full((3,), float('inf'))
# rot_max = torch.full((3,), float('-inf'))

# # 加载PT文件
# data = torch.load(pt_file_path)
# data = data['metadata']

# def update_min_max(data):
#     global joint_angle_min, joint_angle_max, trans_min, trans_max, rot_min, rot_max
    
#     for entry in data:
#         translations = entry['translations']
#         rotations = entry['rotations']
#         joint_positions = entry['joint_positions']

#         # 将 translation 乘以逆旋转矩阵
#         adjusted_translations = translations 


#         # 更新平移
#         trans_min = torch.min(trans_min, adjusted_translations)
#         trans_max = torch.max(trans_max, adjusted_translations)


# # 处理数据
# update_min_max(data)

# # 按指定格式打印结果
# print("_joint_angle_lower = torch.tensor([{}])".format(", ".join(map(str, joint_angle_min.tolist()))))
# print("_joint_angle_upper = torch.tensor([{}])".format(", ".join(map(str, joint_angle_max.tolist()))))
# print("_global_trans_lower = torch.tensor([{}])".format(", ".join(map(str, trans_min.tolist()))))
# print("_global_trans_upper = torch.tensor([{}])".format(", ".join(map(str, trans_max.tolist()))))
# print("_global_rot_lower = torch.tensor([{}])".format(", ".join(map(str, rot_min.tolist()))))
# print("_global_rot_upper = torch.tensor([{}])".format(", ".join(map(str, rot_max.tolist()))))
# import os
# import torch
# import numpy as np
# import transforms3d

# # PT文件路径
# pt_file_path = '/inspurfs/group/mayuexin/datasets/DexGraspNet/dexgraspnet_shadowhand_downsample.pt'

# # Joint names
# joint_names = [
#     'robot0:FFJ3', 'robot0:FFJ2', 'robot0:FFJ1', 'robot0:FFJ0',
#     'robot0:MFJ3', 'robot0:MFJ2', 'robot0:MFJ1', 'robot0:MFJ0',
#     'robot0:RFJ3', 'robot0:RFJ2', 'robot0:RFJ1', 'robot0:RFJ0',
#     'robot0:LFJ4', 'robot0:LFJ3', 'robot0:LFJ2', 'robot0:LFJ1', 'robot0:LFJ0',
#     'robot0:THJ4', 'robot0:THJ3', 'robot0:THJ2', 'robot0:THJ1', 'robot0:THJ0'
# ]

# # List to store all values
# joint_angles = []
# translations_list = []
# rotations_list = []

# # 加载PT文件
# data = torch.load(pt_file_path)
# data = data['metadata']

# # Collect data
# for entry in data:
#     translations = entry['translations']
#     translations_list.append(translations)

# # Convert lists to tensors

# translations_list = torch.stack(translations_list)


# # Calculate mean and std

# translations_mean = translations_list.mean(dim=0)
# translations_std = translations_list.std(dim=0)



# translations_lower = translations_mean - 3 * translations_std
# translations_upper = translations_mean + 3* translations_std


# # # 计算下界和上界
# # joint_angles_lower = joint_angles_mean - joint_angles_std
# # joint_angles_upper = joint_angles_mean + joint_angles_std
# # translations_lower = translations_mean - translations_std
# # translations_upper = translations_mean + translations_std
# # rotations_lower = rotations_mean - rotations_std
# # rotations_upper = rotations_mean + rotations_std


# print("_global_trans_mean = torch.tensor([{}])".format(", ".join(map(str, translations_mean.tolist()))))
# print("_global_trans_std = torch.tensor([{}])".format(", ".join(map(str, translations_std.tolist()))))
# print("_global_trans_lower = torch.tensor([{}])".format(", ".join(map(str, translations_lower.tolist()))))
# print("_global_trans_upper = torch.tensor([{}])".format(", ".join(map(str, translations_upper.tolist()))))
# import torch

# # 定义两个张量
# _joint_angle_lower1 = torch.tensor([-0.5235988, -0.7853982, -0.28088030219078064, -0.05893908441066742, 0.21049419045448303, -0.11322182416915894, -0.264189213514328,
#                                     -0.051349759101867676, 0.26139411330223083, -0.08918479084968567, -0.2583976089954376, -0.02682223916053772,
#                                     0.25034967064857483, -0.09140877425670624, 0.03980225324630737, -0.3140914738178253, -0.03026483952999115,
#                                     0.23421019315719604, -0.10111698508262634, -0.0702691376209259, 0.8040796518325806, -0.20523135364055634, 
#                                     -0.5052621364593506, -0.30270588397979736])

# _joint_angle_lower2 = torch.tensor([-0.5235988, -0.7853982, -0.21832510828971863, -0.0033132582902908325, 0.2037600874900818, -0.1306227743625641, -0.1835252195596695,
#                                     0.010280296206474304, 0.2528662383556366, -0.133841410279274, -0.20444758236408234, 0.06201350688934326, 
#                                     0.2701760530471802, -0.12450878322124481, 0.11788338422775269, -0.29409295320510864, 0.060505181550979614, 
#                                     0.22190874814987183, -0.12498503923416138, -0.016971051692962646, 0.8891584277153015, -0.15773186087608337, 
#                                     -0.5730829238891602, -0.31697356700897217])

# # 逐一比较并选取较小值
# _joint_angle_lower = torch.min(_joint_angle_lower1, _joint_angle_lower2)
# _joint_angle_upper = torch.max(_joint_angle_lower1, _joint_angle_lower2)
# print(_joint_angle_lower)
# print(_joint_angle_upper)
import os
import torch
import numpy as np
import transforms3d

# PT 文件路径
pt_file_path = '/inspurfs/group/mayuexin/datasets/DexGRAB/0.01filter_shadowhand_downsample.pt'

# 加载 PT 文件
data = torch.load(pt_file_path)
data = data['metadata']

# 存储翻译、旋转（仅前两列）和关节角度数据
translations_list = []
rotations_list = []
joint_angles = []

# 收集数据
for entry in data:
    translations = entry['translations'].clone().detach()
    # 提取 rotations 的前两列并展平
    rotations = entry['rotations'].clone().detach()[:, :2].reshape(-1)  # 将前两列展平为一个向量
    joints = entry['joint_positions'].clone().detach()[2:]
    
    translations_list.append(translations)
    rotations_list.append(rotations)
    joint_angles.append(joints)

# 将列表转换为张量
translations_list = torch.stack(translations_list)
rotations_list = torch.stack(rotations_list)
joint_angles = torch.stack(joint_angles)

# 计算每个分量的均值和标准差
translations_mean = translations_list.mean(dim=0)
translations_std = translations_list.std(dim=0)
rotations_mean = rotations_list.mean(dim=0)
rotations_std = rotations_list.std(dim=0)
joint_angles_mean = joint_angles.mean(dim=0)
joint_angles_std = joint_angles.std(dim=0)

# 计算下界和上界
translations_lower = translations_mean - 3 * translations_std
translations_upper = translations_mean + 3 * translations_std
rotations_lower = rotations_mean - 3 * rotations_std
rotations_upper = rotations_mean + 3 * rotations_std
joint_angles_lower = joint_angles_mean - 3 * joint_angles_std
joint_angles_upper = joint_angles_mean + 3 * joint_angles_std

# 计算每组的总均值和总标准差
translations_total_mean = translations_list.view(-1).mean()
translations_total_std = translations_list.view(-1).std()
rotations_total_mean = rotations_list.view(-1).mean()
rotations_total_std = rotations_list.view(-1).std()
joint_angles_total_mean = joint_angles.view(-1).mean()
joint_angles_total_std = joint_angles.view(-1).std()

# 拼接所有数据用于计算整体均值和标准差
all_data = torch.cat([translations_list, rotations_list, joint_angles], dim=1)
overall_mean = all_data.mean(dim=0)
overall_std = all_data.std(dim=0)
overall_lower = overall_mean - 3 * overall_std
overall_upper = overall_mean + 3 * overall_std

overall_total_mean = all_data.view(-1).mean()
overall_total_std = all_data.view(-1).std()

# 打印结果
print("_global_trans_mean = torch.tensor([{}])".format(", ".join(map(str, translations_mean.tolist()))))
print("_global_trans_std = torch.tensor([{}])".format(", ".join(map(str, translations_std.tolist()))))
print("_global_trans_lower = torch.tensor([{}])".format(", ".join(map(str, translations_lower.tolist()))))
print("_global_trans_upper = torch.tensor([{}])".format(", ".join(map(str, translations_upper.tolist()))))
print("translations_total_mean =", translations_total_mean.item())
print("translations_total_std =", translations_total_std.item())

print("_global_rot_mean = torch.tensor([{}])".format(", ".join(map(str, rotations_mean.tolist()))))
print("_global_rot_std = torch.tensor([{}])".format(", ".join(map(str, rotations_std.tolist()))))
print("_global_rot_lower = torch.tensor([{}])".format(", ".join(map(str, rotations_lower.tolist()))))
print("_global_rot_upper = torch.tensor([{}])".format(", ".join(map(str, rotations_upper.tolist()))))
print("rotations_total_mean =", rotations_total_mean.item())
print("rotations_total_std =", rotations_total_std.item())

print("_joint_angles_mean = torch.tensor([{}])".format(", ".join(map(str, joint_angles_mean.tolist()))))
print("_joint_angles_std = torch.tensor([{}])".format(", ".join(map(str, joint_angles_std.tolist()))))
print("_joint_angles_lower = torch.tensor([{}])".format(", ".join(map(str, joint_angles_lower.tolist()))))
print("_joint_angles_upper = torch.tensor([{}])".format(", ".join(map(str, joint_angles_upper.tolist()))))
print("joint_angles_total_mean =", joint_angles_total_mean.item())
print("joint_angles_total_std =", joint_angles_total_std.item())

print("_overall_mean = torch.tensor([{}])".format(", ".join(map(str, overall_mean.tolist()))))
print("_overall_std = torch.tensor([{}])".format(", ".join(map(str, overall_std.tolist()))))
print("_overall_lower = torch.tensor([{}])".format(", ".join(map(str, overall_lower.tolist()))))
print("_overall_upper = torch.tensor([{}])".format(", ".join(map(str, overall_upper.tolist()))))
print("rotations_total_mean = {:.2f}".format(rotations_total_mean.item()))
print("rotations_total_std = {:.2f}".format(rotations_total_std.item()))
print("translations_total_mean = {:.2f}".format(translations_total_mean.item()))
print("translations_total_std = {:.2f}".format(translations_total_std.item()))
print("joint_angles_total_mean = {:.2f}".format(joint_angles_total_mean.item()))
print("joint_angles_total_std = {:.2f}".format(joint_angles_total_std.item()))
print("overall_total_mean = {:.2f}".format(overall_total_mean.item()))
print("overall_total_std = {:.2f}".format(overall_total_std.item()))
