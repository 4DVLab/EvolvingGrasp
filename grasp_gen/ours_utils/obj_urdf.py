# import os

# def generate_urdf(obj_file):
#     obj_name = os.path.splitext(os.path.basename(obj_file))[0]
#     urdf_content = f"""<?xml version="1.0"?>
# <robot name="object">
#   <link name="object">
#     <visual>
#       <origin xyz="0.0 0.0 0.0"/>
#       <geometry>
#         <mesh filename="{obj_file}" scale="1.00 1.00 1.00"/>
#       </geometry>
#     </visual>
#     <collision>
#       <origin xyz="0.0 0.0 0.0"/>
#       <geometry>
#         <mesh filename="{obj_file}" scale="1.00 1.00 1.00"/>
#       </geometry>
#     </collision>
#   </link>
# </robot>
# """
#     urdf_filename = os.path.join(os.path.dirname(obj_file), obj_name + ".urdf")
#     with open(urdf_filename, 'w') as urdf_file:
#         urdf_file.write(urdf_content)
#     print(f"Generated URDF for {obj_file}: {urdf_filename}")

# def process_directory(directory):
#     for root, _, files in os.walk(directory):
#         for file in files:
#             if file.endswith('.ply'):
#                 obj_file = os.path.join(root, file)
#                 generate_urdf(obj_file)

# # 使用示例
# directory = '/inspurfs/group/mayuexin/datasets/DexGRAB/contact_meshes'
# process_directory(directory)
import os

def generate_urdf(obj_file):
    obj_name = os.path.splitext(os.path.basename(obj_file))[0]
    urdf_content = f"""<?xml version="1.0"?>
<robot name="object">
  <link name="object">
    <visual>
      <origin xyz="0.0 0.0 0.0"/>
      <geometry>
        <mesh filename="{obj_file}" scale="1.00 1.00 1.00"/>
      </geometry>
    </visual>
    <collision>
      <origin xyz="0.0 0.0 0.0"/>
      <geometry>
        <mesh filename="{obj_file}" scale="1.00 1.00 1.00"/>
      </geometry>
    </collision>
  </link>
</robot>
"""
    urdf_filename = os.path.join(os.path.dirname(obj_file), obj_name + ".urdf")
    with open(urdf_filename, 'w') as urdf_file:
        urdf_file.write(urdf_content)
    print(f"Generated URDF for {obj_file}: {urdf_filename}")

def process_directory(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.off'):
                obj_file = os.path.join(root, file)
                generate_urdf(obj_file)

# 使用示例
directory = '/inspurfs/group/mayuexin/datasets/scaled_PUnetdata/mesh'
process_directory(directory)
