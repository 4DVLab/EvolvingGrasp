import numpy as np

def npy_to_obj(npy_file, obj_file):
    # Load the .npy file
    data = np.load(npy_file, allow_pickle=True)
    
    # Open the .obj file for writing
    with open(obj_file, 'w') as f:
        # Write vertices
        for point in data[5]:
            f.write(f'v {point[0]} {point[1]} {point[2]}\n')

# Convert pc.npy to pc.obj
npy_file = '/inspurfs/group/mayuexin/datasets/UniDexGrasp/DFCData/meshdatav3/core/bottle-1a7ba1f4c892e2da30711cdbdbc73924/pcs_table.npy'
obj_file = 'pc.obj'
npy_to_obj(npy_file, obj_file)
print(f"Converted {npy_file} to {obj_file}")
