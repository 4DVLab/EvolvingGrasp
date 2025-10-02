# import torch
# import os
# from datetime import datetime

# def merge_pt_files(source_dir, output_file, after_time):
#     """
#     Merges multiple .pt files from a directory into a single .pt file, combining 'metadata' entries.

#     Args:
#     - source_dir (str): Directory containing the .pt files.
#     - output_file (str): Path to save the merged .pt file.
#     - after_time (datetime): Only merge files modified after this time.
#     """
#     merged_metadata = []
#     info_data = None  # Initialize to None, will hold the last info_data encountered

#     # Iterate through all files in the source directory
#     for filename in os.listdir(source_dir):
#         if filename.endswith('.pt'):  # Check for .pt files
#             file_path = os.path.join(source_dir, filename)
            
#             # Check file's last modified time
#             file_mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
#             if file_mtime > after_time:  # Only process files modified after the specified time
#                 # Load the data from the .pt file
#                 print(f"Processing {file_path} modified at {file_mtime}")
#                 data = torch.load(file_path)
#                 if 'metadata' in data:
#                     print('merging metadata')
#                     merged_metadata.extend(data['metadata'])  # Append all metadata to the list
#                 if 'info' in data:
#                     info_data = data['info']  # Optionally update info_data with the last one found

#     # Prepare the final dictionary to save
#     final_data = {
#         'info': info_data,  # This assumes all files had consistent 'info' or you just need the last one
#         'metadata': merged_metadata
#     }

#     # Save the merged data into a single .pt file
#     torch.save(final_data, output_file)
#     print(f'Merged data saved to {output_file}')

# # Example usage
# source_directory = '/inspurfs/group/mayuexin/datasets/grasp_anyting/mesh1_pt'  # Set the path to your .pt files
# output_path = '/inspurfs/group/mayuexin/datasets/grasp_anyting/8_22_30-merged_data.pt'  # Set the path to save merged file
# after_time = datetime(2024, 10, 1, 22, 30)  # Only include files modified after this date and time
# merge_pt_files(source_directory, output_path, after_time)
import os
import torch
import json
from collections import defaultdict
from datetime import datetime

def merge_pt_files(source_dir, output_file, after_time):
    merged_metadata = []
    info_data = None  # 用于存储最后一个 info 数据

    for filename in os.listdir(source_dir):
        if filename.endswith('.pt'):
            file_path = os.path.join(source_dir, filename)
            file_mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
            if file_mtime > after_time:
                print(f"Processing {file_path} modified at {file_mtime}")
                data = torch.load(file_path)
                if 'metadata' in data:
                    merged_metadata.extend(data['metadata'])
                if 'info' in data:
                    info_data = data['info']

    final_data = {
        'info': info_data,
        'metadata': merged_metadata
    }

    torch.save(final_data, output_file)
    print(f'Merged data saved to {output_file}')

def deduplicate_metadata(input_file, output_file):
    data = torch.load(input_file)
    metadata = data['metadata']

    unique_translations = {}
    filtered_metadata = []

    for entry in metadata:
        trans_tuple = tuple(entry['translations'].tolist())
        if trans_tuple not in unique_translations:
            unique_translations[trans_tuple] = entry
            filtered_metadata.append(entry)

    data['metadata'] = filtered_metadata
    torch.save(data, output_file)

    print(f"去重后数据条目数: {len(filtered_metadata)}")

def calculate_info_counts(input_file):
    data = torch.load(input_file)
    metadata = data.get('metadata', [])

    object_counts = defaultdict(int)

    for object_data in metadata:
        object_name = object_data['object_name']
        object_counts[object_name] += 1

    total_count = sum(object_counts.values())
    data['info']['num_per_object'] = dict(object_counts)
    data['info']['num_total'] = total_count

    torch.save(data, input_file)
    print("info 文件内容已更新")

def save_objects_above_count(input_file, output_json_file, min_count=200):
    data = torch.load(input_file)
    num_per_object = data['info']['num_per_object']
    objects_above_count = {obj: count for obj, count in num_per_object.items() if count > min_count}

    with open(output_json_file, 'w') as json_file:
        json.dump(objects_above_count, json_file, indent=4)

    print(f"筛选结果已保存到 {output_json_file}")

# 示例调用
source_directory = '/inspurfs/group/mayuexin/datasets/grasp_anyting/mesh1_pt'
merged_file_path = '/inspurfs/group/mayuexin/datasets/grasp_anyting/10_9_0_24merged_file.pt'
after_time = datetime(2024, 10, 1, 22, 30)

# 执行合并文件操作
merge_pt_files(source_directory, merged_file_path, after_time)

# 去重操作
deduplicate_metadata(merged_file_path, merged_file_path)

# 计算 info 信息
calculate_info_counts(merged_file_path)

# 保存超过 200 个物体的 JSON 文件
output_json_path = '/inspurfs/group/mayuexin/datasets/grasp_anyting/objects_above_200.json'
save_objects_above_count(merged_file_path, output_json_path, min_count=200)
