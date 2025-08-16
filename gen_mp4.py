import pyvista as pv
import glob
import imageio.v2 as imageio  # use v2 API
import os

# ---------------------------
# User configuration
# ---------------------------
vtu_dir = "vtk_outputs"           # Folder with .vtu files
scalar_name = "Velocity"          # Scalar to color by (must match your VTU arrays)
output_mp4 = "inc-cylinder.mp4"   # Output video
fps = 24                          # Frames per second
cmap = "viridis"                  # Color map
off_screen = True                 # Render offscreen
window_size = (1920, 1080)        # Resolution of rendered frames (width, height)

# ---------------------------
# Load all VTU files
# ---------------------------
vtu_files = sorted(glob.glob(os.path.join(vtu_dir, "inc-cylinder_*.vtu")))
if not vtu_files:
    raise FileNotFoundError(f"No VTU files found in {vtu_dir}")

grids = [pv.read(f) for f in vtu_files]

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
    plotter.add_mesh(grid, scalars=scalar_name, cmap=cmap, show_edges=False)
    
    # Optional: adjust camera (example: top view)
    plotter.camera_position = 'xy'
    
    # Render offscreen and save as image
    img_path = f"frame_{i:04d}.png"
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
for img_path in glob.glob("frame_*.png"):
    os.remove(img_path)
