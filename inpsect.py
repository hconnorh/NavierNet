# inspect_pyfrs.py
# import h5py
# import sys
# from pprint import pprint

# fname = sys.argv[1] if len(sys.argv)>1 else 'inc-cylinder-75.00.pyfrs'
# with h5py.File(fname, 'r') as f:
#     def walk(g, prefix=''):
#         for k in g.keys():
#             item = g[k]
#             path = f"{prefix}/{k}"
#             if isinstance(item, h5py.Dataset):
#                 print(f"DATASET: {path} shape={item.shape} dtype={item.dtype}")
#             else:
#                 print(f"GROUP:   {path}")
#                 walk(item, path)
#     walk(f)

import h5py
import numpy as np
import matplotlib.pyplot as plt

with h5py.File("inc-cylinder-75.00.pyfrs", "r") as f:
    data = f["/soln_tri_p0"][...]   # shape = (n_elements, n_vars, n_points)
    print("Shape:", data.shape)

    # Example: take pressure (index 0), flatten
    pressure = data[:,0,:].flatten()

    plt.hist(pressure, bins=50)
    plt.title("Pressure distribution")
    plt.show()
