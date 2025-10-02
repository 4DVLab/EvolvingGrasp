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
                # print(f"Processing {file_path} modified at {file_mtime}")
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
    # print(f'Merged data saved to {output_file}')

def deduplicate_metadata(input_file, output_file):
    data = torch.load(input_file)
    metadata = data['metadata']

    unique_translations = {}
    filtered_metadata = []

    for entry in metadata:
        # trans_tuple = tuple(entry['translations'].tolist())
        trans_tuple = tuple(entry['rotations'][0].tolist())
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
    # print(data['info']['num_per_object'])
    torch.save(data, input_file)
    print("info 文件内容已更新")

def save_objects_above_count(input_file, output_json_file, min_count=200):
    # 加载数据
    data = torch.load(input_file)
    num_per_object = data['info']['num_per_object']
    
    # 筛选并排序对象，按照数量从高到底排序
    objects_above_count = {obj: count for obj, count in num_per_object.items() if count > min_count}
    objects_above_count = dict(sorted(objects_above_count.items(), key=lambda item: item[1], reverse=True))
    
    # 输出筛选后的物体数量
    print(f"筛选出的物体数量: {len(objects_above_count)}")

    # 保存到JSON文件
    with open(output_json_file, 'w') as json_file:
        json.dump(objects_above_count, json_file, indent=4)

    print(f"筛选结果已保存到 {output_json_file}")

# 示例调用
source_directory = '/inspurfs/group/mayuexin/datasets/grasp_anyting/mesh1_pt'
merged_file_path = '/inspurfs/group/mayuexin/datasets/grasp_anyting/10_10_19_35merged_file.pt'
after_time = datetime(2024, 10, 17, 15, 40)
# after_time = datetime(2024, 10, 10, 15, 40)
# 执行合并文件操作
merge_pt_files(source_directory, merged_file_path, after_time)

# 去重操作
deduplicate_metadata(merged_file_path, merged_file_path)

# 计算 info 信息
calculate_info_counts(merged_file_path)

# 保存超过 200 个物体的 JSON 文件
output_json_path = '/inspurfs/group/mayuexin/datasets/grasp_anyting/objects_above_200.json'
save_objects_above_count(merged_file_path, output_json_path, min_count=0)