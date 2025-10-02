import numpy as np

def load_npz_file(file_path):
    # Load the .npz file
    data = np.load(file_path, allow_pickle=True)
    
    # Extract the contents
    contents = {key: data[key] for key in data.keys()}
    
    # Convert contents to a readable format
    readable_contents = {}
    for key, value in contents.items():
        if isinstance(value, np.ndarray):
            readable_contents[key] = value.tolist()
        else:
            readable_contents[key] = value
    
    return readable_contents

# Example usage
file_path = '/inspurfs/group/mayuexin/datasets/UniDexGrasp/DFCData/datasetv4.1/core/cellphone-5c141e299b50f1139f24fa4acce33c04/00030.npz'  # Replace with your file path
npz_contents = load_npz_file(file_path)
print(npz_contents)
# # Print the contents
# for key, value in npz_contents.items():
#     print(f"{key}: {value}")
