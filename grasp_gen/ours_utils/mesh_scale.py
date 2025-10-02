import open3d as o3d
import numpy as np
import os

def compute_mesh_size(mesh):
    # 获取mesh的最小和最大边界
    min_bound = mesh.get_min_bound()
    max_bound = mesh.get_max_bound()
    # 计算尺寸
    return max_bound - min_bound

def scale_mesh(mesh, target_size):
    # 计算当前mesh的大小
    current_size = compute_mesh_size(mesh)
    # 计算缩放因子，保持比例缩放
    scale_factor = target_size / max(current_size)
    # 缩放mesh
    mesh.scale(scale_factor*0.75, center=mesh.get_center())
    return mesh

def process_mesh_files(input_dir, output_dir):
    # 获取目录下所有的mesh文件
    mesh_files = [f for f in os.listdir(input_dir) if f.endswith('.off') or f.endswith('.obj')]
    
    # 确定目标大小（使用第一个mesh作为基准）
    first_mesh = o3d.io.read_triangle_mesh("/inspurfs/group/mayuexin/datasets/Realdex/meshdata/box.obj")
    target_size = max(compute_mesh_size(first_mesh))

    # 逐个处理每个mesh文件
    for mesh_file in mesh_files:
        # 读取mesh
        mesh = o3d.io.read_triangle_mesh(os.path.join(input_dir, mesh_file))
        
        # 缩放mesh到目标大小
        scaled_mesh = scale_mesh(mesh, target_size)
        
        # 保存缩放后的mesh到输出目录
        output_file = os.path.join(output_dir, mesh_file)
        o3d.io.write_triangle_mesh(output_file, scaled_mesh)
        print(f"Processed and saved: {output_file}")

# 使用示例
input_dir = "/inspurfs/group/mayuexin/datasets/scaled_PUnetdata/mesh"
output_dir = "/inspurfs/group/mayuexin/datasets/scaled_PUnetdata/mesh"
os.makedirs(output_dir, exist_ok=True)

process_mesh_files(input_dir, output_dir)
