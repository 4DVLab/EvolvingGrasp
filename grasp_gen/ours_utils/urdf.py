from pytorch_kinematics.urdf_parser_py.urdf import (URDF, Box, Cylinder, Mesh, Sphere)
import torch
import numpy as np
# URDF 文件路径
urdf_filename = '/inspurfs/group/mayuexin/zym/diffusion+hand/Scene-Diffuser/assets/urdf/sr_grasp_description/urdf/shadowhand.urdf'

# 读取并解析 URDF 文件，生成 URDF 对象
visual = URDF.from_xml_string(open(urdf_filename).read())

# 使用 pk 库从 URDF 文件中构建机器人的运动链，并转换为 PyTorch 张量
# 假设 pk 是一个支持从 URDF 构建运动链的库
import pytorch_kinematics as pk
robot = pk.build_chain_from_urdf(open(urdf_filename).read()).to(dtype=torch.float)

# 使用 URDF_PARSER 库读取并解析 URDF 文件
import urdf_parser_py.urdf as URDF_PARSER
robot_full = URDF_PARSER.URDF.from_xml_file(urdf_filename)
for i_link, link in enumerate(visual.links):
    # print(f"Processing link #{i_link}: {link.name}")
    # load mesh
    if len(link.visuals) == 0:
        continue
    print(link.name)
    print(type(link.visuals[0].geometry))
# 打印解析结果
# print(visual._root)
# print("Links:", [link.name for link in visual.links])
# print("Joints:", [joint.name for joint in visual.joints])
# print(robot)
# print(robot_full)



