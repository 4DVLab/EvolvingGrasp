
import numpy as np

def load_npy_file(file_path):
    try:
        data = np.load(file_path, allow_pickle=True)
        print(f"Type: {type(data)}")
        print(f"Shape: {data}")
        #print(data)
        # 如果是零维数组，直接打印其内容
        if data.shape == ():
            content = data.item()
            print(f"Content: {content}")
            # 如果内容是字典，打印其键和值
            if isinstance(content, dict):
                print("Keys:", content.keys())
                for key, value in content.items():
                    if isinstance(value, np.ndarray):
                        print(f"{key}: Type: {type(value)}, Shape: {value.shape}, Sample: {value[111]}")
                    else:
                        print(f"{key}: Type: {type(value)}, Sample: {value}")
            else:
                print(f"Content: {content}")
                
        # 如果是非零维数组，打印样本内容
        elif data.ndim > 0:
            print(f"Content Sample: {data[:5]}")
            
    except Exception as e:
        print(f"Error loading {file_path}: {e}")

file_path1 = "/inspurfs/group/mayuexin/zym/diffusion+hand/ugg/data/unidex/ZX700_mzGbdP3u6JB.npy"
load_npy_file(file_path1)