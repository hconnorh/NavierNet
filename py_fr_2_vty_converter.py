import subprocess
import glob
import os

mesh_file = "inc-cylinder.pyfrm"
out_dir = "vtk_outputs"
os.makedirs(out_dir, exist_ok=True)

# Find all .pyfrs files and sort
pyfrs_files = sorted(glob.glob("inc-cylinder-*.pyfrs"))

# Prepare PVD entries
pvd_entries = []

for idx, sol_file in enumerate(pyfrs_files):
    timestep = float(sol_file.split("-")[-1].split(".")[0])  # extract time from filename
    out_file = os.path.join(out_dir, f"inc-cylinder_{idx:04d}.vtu")
    
    print(f"Converting {sol_file} -> {out_file}")
    subprocess.run(["pyfr", "export", mesh_file, sol_file, out_file])
    
    # Add entry for PVD
    pvd_entries.append((timestep, os.path.basename(out_file)))

# Write PVD file
pvd_path = os.path.join(out_dir, "inc-cylinder.pvd")
with open(pvd_path, "w") as f:
    f.write('<?xml version="1.0"?>\n')
    f.write('<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n')
    f.write('  <Collection>\n')
    for timestep, filename in pvd_entries:
        f.write(f'    <DataSet timestep="{timestep}" group="" part="0" file="{filename}"/>\n')
    f.write('  </Collection>\n')
    f.write('</VTKFile>\n')

print(f"\nDone! PVD file created at: {pvd_path}")
print("Open this PVD in ParaView to animate your simulation.")
