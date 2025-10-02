import torch

# 加载原始的 .pt 文件
input_file = '/inspurfs/group/mayuexin/datasets/grasp_anyting/8_22_30merged_data_unique.pt'  # 替换为你的原始 pt 文件路径
output_file = '/inspurfs/group/mayuexin/datasets/grasp_anyting/8_22_30merged_data_unique.pt'  # 新生成的文件名

# 加载数据
data = torch.load(input_file)

# 获取 metadata
metadata = data['metadata']

# 使用字典来存储唯一的 translations
unique_translations = {}
filtered_metadata = []

for entry in metadata:
    # 将 translations 转换为 tuple 以便比较
    trans_tuple = tuple(entry['rotations'][0].tolist())
    
    # 如果该 translation 不存在，添加到字典和过滤后的列表
    if trans_tuple not in unique_translations:
        unique_translations[trans_tuple] = entry
        filtered_metadata.append(entry)

# 将去重后的 metadata 替换原始的 metadata
data['metadata'] = filtered_metadata

# 保存为新的 .pt 文件
torch.save(data, output_file)

print(f"原始数据条目数: {len(metadata)}")
print(f"去重后数据条目数: {len(filtered_metadata)}")
