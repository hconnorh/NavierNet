import pyvista as pv
import numpy as np
import glob
import imageio.v2 as imageio  # use v2 API
import os
from pathlib import Path
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent

def gen_mp4(sim_name, case_name, remove_images=True, fps=20, cmap="viridis", off_screen=True, window_size=(1920, 1080)):
    """
    Generate an MP4 animation of velocity magnitude from a sequence of .vtu files.

    Args:
        sim_name (str): The simulation name (used as a directory under "sims/").
        case_name (str): The case name (directory under "pyfr_results").
        remove_images (bool): Whether to remove the temporary frames after making video.

    Output:
        An MP4 video is saved to sims/{sim_name}/animations/{sim_name}-{case_name}.mp4
    """

    sim_dir = ROOT / "sims" / sim_name
    vtu_dir = sim_dir / "pyfr_results" / case_name
    anim_dir = sim_dir / "animations"
    anim_dir.mkdir(parents=True, exist_ok=True)

    scalar_name = "Velocity" # Vector field to plot Euclidean norm of
    output_mp4 = anim_dir / f"{sim_name}-{case_name}.mp4"


    # Load all VTU files
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

    # Prepare plotter
    plotter = pv.Plotter(off_screen=off_screen, window_size=window_size)
    
    # Render each timestep
    frames = []
    for i, grid in enumerate(tqdm(grids)):
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
        img_path = anim_dir / f"frame_{i:04d}.png"
        plotter.screenshot(str(img_path))
        frames.append(imageio.imread(str(img_path)))
        # print(f"Rendered frame {i+1}/{len(grids)}")

    # Save animation
    imageio.mimsave(output_mp4, frames, fps=fps)
    print(f"Animation saved to {output_mp4}")

    # Cleanup images
    if remove_images:
        for img_path in glob.glob("frame_*.png"):
            os.remove(img_path)


if __name__ == "__main__":
    sim_name = "test2"
    fps = 20                                   # Frames per second
    cmap = "viridis"                           # Color map
    off_screen = True                          # Render offscreen
    window_size = (1920, 1080)                 # Resolution of rendered frames (width, height)

    gen_mp4(sim_name, "case0", remove_images=True, fps=fps, cmap=cmap, off_screen=off_screen, window_size=window_size)