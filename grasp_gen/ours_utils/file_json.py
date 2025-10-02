# # # import os
# # # import random
# # # import json

# # # def split_files(folder_path, train_ratio=0.8):
# # #     # 获取文件夹中的所有文件名称
# # #     # all_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
# # #     # 文件夹名称
# # #     all_files = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]

# # #     # 打乱文件顺序
# # #     random.shuffle(all_files)
    
# # #     # 计算训练和测试集的大小
# # #     train_size = int(len(all_files) * train_ratio)
    
# # #     # 分割文件
# # #     train_files = all_files[:train_size]
# # #     test_files = all_files[train_size:]
    
# # #     return train_files, test_files, all_files

# # # def save_to_json(train_files, test_files, all_files, output_file):
# # #     data = {
# # #         "_train_split": train_files,
# # #         "_test_split": test_files,
# # #         "_all_split": all_files
# # #     }
# # #     with open(output_file, 'w') as f:
# # #         json.dump(data, f, indent=4)

# # # # 使用示例
# # # folder_path = "/inspurfs/group/mayuexin/datasets/DexGraspNet/meshdata"
# # # output_file = "/inspurfs/group/mayuexin/datasets/DexGraspNet/mesh.json"
# # # train_files, test_files, all_files = split_files(folder_path)
# # # save_to_json(train_files, test_files, all_files, output_file)

# # # print(f"分割结果已保存到 {output_file}")
# # # import os
# # # import json

# # # def load_files(file_path):
# # #     with open(file_path, 'r') as f:
# # #         files = [line.strip() for line in f.readlines()]
# # #     return files

# # # def save_to_json(train_files, test_files, output_file):
# # #     data = {
# # #         "_train_split": train_files,
# # #         "_test_split": test_files,
# # #         "_all_split": train_files + test_files
# # #     }
# # #     with open(output_file, 'w') as f:
# # #         json.dump(data, f, indent=4)

# # # # 使用示例
# # # train_file_path = "/inspurfs/group/mayuexin/zym/diffusion+hand/Scene-Diffuser/train.txt"
# # # test_file_path = "/inspurfs/group/mayuexin/zym/diffusion+hand/Scene-Diffuser/test.txt"
# # # output_file = "/inspurfs/group/mayuexin/datasets/DexGraspNet/grasp.json"

# # # train_files = load_files(train_file_path)
# # # test_files = load_files(test_file_path)

# # # save_to_json(train_files, test_files, output_file)

# # # print(f"分割结果已保存到 {output_file}")
# # # import os
# # # import json

# # # def merge_json_files(input_folder, output_file):
# # #     merged_data = {
# # #         "_train_split": [],
# # #         "_test_split": [],
# # #         "_all_split": []
# # #     }

# # #     # 遍历文件夹中的所有 JSON 文件
# # #     for filename in os.listdir(input_folder):
# # #         if filename.endswith(".json"):
# # #             file_path = os.path.join(input_folder, filename)
# # #             with open(file_path, 'r') as f:
# # #                 data = json.load(f)
# # #                 if "train" in data:
# # #                     merged_data["_train_split"].extend(data["train"])
# # #                 if "test" in data:
# # #                     merged_data["_test_split"].extend(data["test"])

# # #     # 将所有文件路径合并到 _all_split 中
# # #     merged_data["_all_split"] = merged_data["_train_split"] + merged_data["_test_split"]

# # #     # 保存合并后的数据到新的 JSON 文件
# # #     with open(output_file, 'w') as f:
# # #         json.dump(merged_data, f, indent=4)

# # #     print(f"Merged data saved to {output_file}")

# # # # 使用示例
# # # input_folder = "/inspurfs/group/mayuexin/datasets/UniDexGrasp/DFCData/splits"  # 替换为你的 JSON 文件所在文件夹路径
# # # output_file = "/inspurfs/group/mayuexin/datasets/UniDexGrasp/DFCData/grasp.json"  # 替换为你希望保存合并后 JSON 文件的路径

# # # merge_json_files(input_folder, output_file)

# # import os
# # import random
# # import json

# # def split_folders(folder_path, test_folders_fixed, train_ratio=0.8):
# #     # 获取文件夹中的所有子文件夹名称
# #     all_folders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]

# #     # 确保固定测试集文件夹在所有文件夹中
# #     test_folders = [f for f in test_folders_fixed if f in all_folders]

# #     # 剔除测试集文件夹后的剩余文件夹
# #     remaining_folders = [f for f in all_folders if f not in test_folders]

# #     # 打乱剩余文件夹顺序
# #     random.shuffle(remaining_folders)
    
# #     # 计算训练集的大小
# #     train_size = int(len(remaining_folders) * train_ratio)
    
# #     # 分割文件夹
# #     train_folders = remaining_folders[:train_size]
# #     test_folders += remaining_folders[train_size:]  # 将剩余文件夹也添加到测试集中
    
# #     return train_folders, test_folders, all_folders

# # def save_to_json(train_folders, test_folders, all_folders, output_file):
# #     data = {
# #         "_train_split": train_folders,
# #         "_test_split": test_folders,
# #         "_all_split": all_folders
# #     }
# #     with open(output_file, 'w') as f:
# #         json.dump(data, f, indent=4)

# # # 使用示例
# # folder_path = "/inspurfs/group/mayuexin/datasets/scaled_PUnetdata/mesh"
# # output_file = "/inspurfs/group/mayuexin/datasets/scaled_PUnetdata/grasp.json"
# # test_folders_fixed = ['goji_jar', 'small_sprayer', 'yogurt', 'body_lotion', 'bowling_game_box', 'chips', 'duck_toy', 'cosmetics', 'sprayer', 'box', 'daily_moisture_lotion', 'midew_remover']

# # train_folders, test_folders, all_folders = split_folders(folder_path, test_folders_fixed)
# # save_to_json(train_folders, test_folders, all_folders, output_file)

# # print(f"分割结果已保存到 {output_file}")

# # import os
# # import random
# # import json

# # def split_files(folder_path, test_files_fixed, train_ratio=0.8):
# #     # 获取文件夹中的所有文件名称并移除 .off 后缀
# #     all_files = [f[:-4] for f in os.listdir(folder_path) if f.endswith('.off')]

# #     # 确保固定测试集文件在所有文件中
# #     test_files = [f[:-4] for f in test_files_fixed if f[:-4] in all_files]

# #     # 剔除测试集文件后的剩余文件
# #     remaining_files = [f for f in all_files if f not in test_files]

# #     # 打乱剩余文件顺序
# #     random.shuffle(remaining_files)
    
# #     # 计算训练集的大小
# #     train_size = int(len(remaining_files) * train_ratio)
    
# #     # 分割文件
# #     train_files = remaining_files[:train_size]
# #     test_files += remaining_files[train_size:]  # 将剩余文件也添加到测试集中
    
# #     return train_files, test_files, all_files

# # def save_to_json(train_files, test_files, all_files, output_file):
# #     data = {
# #         "_train_split": train_files,
# #         "_test_split": test_files,
# #         "_all_split": all_files
# #     }
# #     with open(output_file, 'w') as f:
# #         json.dump(data, f, indent=4)

# # # 使用示例
# # folder_path = "/inspurfs/group/mayuexin/datasets/scaled_PUnetdata/mesh"
# # output_file = "/inspurfs/group/mayuexin/datasets/scaled_PUnetdata/grasp.json"
# # test_files_fixed = ['goji_jar.off', 'small_sprayer.off', 'yogurt.off', 'body_lotion.off', 'bowling_game_box.off', 
# #                     'chips.off', 'duck_toy.off', 'cosmetics.off', 'sprayer.off', 'box.off', 
# #                     'daily_moisture_lotion.off', 'midew_remover.off']

# # train_files, test_files, all_files = split_files(folder_path, test_files_fixed)
# # save_to_json(train_files, test_files, all_files, output_file)

# # print(f"分割结果已保存到 {output_file}")

# import torch

# # 创建空字典结构，与用户提供的样本匹配
# data_structure = {
#     'info': {
#         'robot_name': 'shadowhand',
#         'num_total': 0,
#         'num_per_object': {}
#     },
#     'metadata': []
# }

# # 将空结构保存为 pt 文件
# torch.save(data_structure, '/inspurfs/group/mayuexin/datasets/scaled_PUnetdata/grasp_anyting_shadowhand_downsample.pt')

