import pyvista as pv
import numpy as np
import glob
import imageio.v2 as imageio  # use v2 API
import os

# ---------------------------
# User configuration
# ---------------------------
sim_name = "water-1200f-3000re"
vtu_dir = f"sims/{sim_name}/pyfr_results/case0"           # Folder with .vtu files
scalar_name = "Velocity_mag"               # Color by velocity magnitude for consistency
output_mp4 = f"{sim_name}-animation.mp4"   # Output video
fps = 20                                   # Frames per second
cmap = "viridis"                           # Color map
off_screen = True                          # Render offscreen
window_size = (1920, 1080)                 # Resolution of rendered frames (width, height)

# ---------------------------
# Load all VTU files
# ---------------------------
vtu_files = sorted(glob.glob(os.path.join(vtu_dir, "inc-cylinder_*.vtu")))
if not vtu_files:
    raise FileNotFoundError(f"No VTU files found in {vtu_dir}")

grids = [pv.read(f) for f in vtu_files]

# Compute velocity magnitude for each grid and establish global color limits
global_min = float("inf")
global_max = float("-inf")
for grid in grids:
    if "Velocity" not in grid.point_data:
        raise KeyError("Velocity array not found in VTU file.")
    vel = grid.point_data["Velocity"]
    mag = np.linalg.norm(vel, axis=1) if vel.ndim > 1 else vel
    grid.point_data["Velocity_mag"] = mag
    vmin = float(mag.min())
    vmax = float(mag.max())
    if vmin < global_min:
        global_min = vmin
    if vmax > global_max:
        global_max = vmax

# ---------------------------
# Prepare plotter
# ---------------------------
plotter = pv.Plotter(off_screen=off_screen, window_size=window_size)
frames = []

# ---------------------------
# Render each timestep
# ---------------------------
for i, grid in enumerate(grids):
    plotter.clear()
    # Add mesh with scalar coloring
    plotter.add_mesh(
        grid,
        scalars=scalar_name,
        cmap=cmap,
        show_edges=False,
        clim=(global_min, global_max),
    )
    
    # Optional: adjust camera (example: top view)
    plotter.camera_position = 'xy'
    
    # Render offscreen and save as image
    img_path = f"sims/{sim_name}/animations/frame_{i:04d}.png"
    plotter.screenshot(img_path)
    frames.append(imageio.imread(img_path))
    print(f"Rendered frame {i+1}/{len(grids)}")

# ---------------------------
# Save animation
# ---------------------------
imageio.mimsave(output_mp4, frames, fps=fps)
print(f"Animation saved to {output_mp4}")

# ---------------------------
# Cleanup images
# ---------------------------
# for img_path in glob.glob("frame_*.png"):
#     os.remove(img_path)
