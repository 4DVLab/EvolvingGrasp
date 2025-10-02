import torch
import pickle
import os
# # 加载 .pt 文件
# pt_file_path = '/inspurfs/group/mayuexin/datasets/DexGraspNet/dexgraspnet_shadowhand_downsample.pt'
# data = torch.load(pt_file_path)

# # 保存为 .pkl 文件
# pkl_file_path = '/inspurfs/group/mayuexin/datasets/DexGraspNet/dexgraspnet_shadowhand_downsample.pkl'
# with open(pkl_file_path, 'wb') as f:
#     pickle.dump(data, f)

# print(f"Data has been successfully saved to {pkl_file_path}")
grasps = pickle.load(open(os.path.join( '/public/home/zym1525742497/outputs/2024-08-10_02-25-17_combined_datasets4_12_6/eval/final/wo1/res_diffuser.pkl'), 'rb'))
print(grasps['sample_qpos']['ycb+tomato_soup_can'][0])