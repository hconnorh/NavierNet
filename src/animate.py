import pyvista as pv
import numpy as np
import glob
import imageio.v2 as imageio  # use v2 API
import os
import re

from pathlib import Path
from tqdm import tqdm
from PIL import Image
from matplotlib.colors import LinearSegmentedColormap

from plots import plot_compare, CMAP

ROOT = Path(__file__).resolve().parent.parent


"""
Module: animate.py
Package: NavierNet
Author: @hconnorh
Description: Utilities for generating animations of the simulation results.
"""


ELECTRIC_COLORMAP_MPL = LinearSegmentedColormap.from_list(
    "electric",
    [
        (0.0, (0.0/255.0, 0.0/255.0, 0.0/255.0)),  # black
        (0.37695419864113106, (2.0/255.0, 16.0/255.0, 83.0/255.0)),
        (0.559445018580004, (9.0/255.0, 59.0/255.0, 157.0/255.0)),
        (0.7490160293761956, (22.0/255.0, 115.0/255.0, 221.0/255.0)),
        (0.9068800985070032, (73.0/255.0, 174.0/255.0, 243.0/255.0)),
        (1.0, (255.0/255.0, 255.0/255.0, 255.0/255.0)),  # white
    ],
    N=256,
)
ELECTRIC_COLORMAP_MPL_R = ELECTRIC_COLORMAP_MPL.reversed()

MPL_CMAP_MAP = {
    "electric": ELECTRIC_COLORMAP_MPL,
    "electric_r": ELECTRIC_COLORMAP_MPL_R,
    "turbo": "turbo",
    "viridis": "viridis",
    # Grayscale with low=white, high=black
    "greys": "Greys",
    "gray_r": "gray_r",
}


def vtu_to_mp4(sim_name, case_name, remove_images=True, fps=20, cmap="viridis", 
               off_screen=True, window_size=(1920, 1080)):
    """
    Generate an MP4 animation of velocity magnitude from a sequence of .vtu 
    files.

    Args:
        sim_name (str): The simulation name (used as a directory under "sims/").
        case_name (str): The case name (directory under "pyfr_results").
        remove_images (bool): Whether to remove the temporary frames after 
        making video.

    Output:
        An MP4 video is saved to sims/{sim_name}/animations/{sim_name}-{case_name}.mp4
    """

    sim_dir = ROOT / "sims" / sim_name
    vtu_dir = sim_dir / "pyfr_results" / case_name
    anim_dir = sim_dir / "animations" / "sim_plots"
    anim_dir.mkdir(parents=True, exist_ok=True)

    scalar_name = "Velocity" # Vector field to plot Euclidean norm of
    output_mp4 = anim_dir.parent / f"sim-results-{sim_name}-{case_name}.mp4"

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
    # Use a clean background and ensure the render area fills the window
    plotter.set_background('white')
    mpl_cmap = MPL_CMAP_MAP.get(cmap, cmap)
    
    # Render each timestep
    frames = []
    for i, grid in enumerate(tqdm(grids)):
        plotter.clear()
        # Add mesh with scalar coloring
        plotter.add_mesh(
            grid,
            scalars="Velocity_mag",  # color by computed magnitude
            cmap=mpl_cmap,
            show_edges=False,
            clim=(global_min, global_max),
            show_scalar_bar=True,
            scalar_bar_args=dict(
                title="",
                vertical=True,
                position_x=0.75,  
                position_y=0.315,
                height=0.36,      
                width=0.05,
                label_font_size=12,      # Make colorbar tick/value labels smaller
                title_font_size=12,
                n_labels=5,             # Fewer labels for clarity (optional)
            ),
        )
        
        # Orient and fit camera so content fills the window and zoom in
        plotter.view_xy()
        plotter.enable_parallel_projection()
        plotter.reset_camera()

        # Leave a small right margin for the colorbar
        try:
            plotter.renderer.SetViewport(0.0, 0.0, 0, 1.0)
        except Exception:
            pass
        
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

def _natural_key(path: str):
    """Sorts order of png's based on numerical naming convention"""
    name = os.path.basename(path)
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]

def _prepare_frame(img: np.ndarray, target_w, target_h) -> np.ndarray:
    """
    Normalise and resize an image for animation frames.
    """

    # Normalise to RGB uint8
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    if img.shape[-1] == 4:
        rgb = img[..., :3].astype(np.float32)
        alpha = (img[..., 3:4].astype(np.float32)) / 255.0
        img = (rgb * alpha + 255.0 * (1.0 - alpha)).astype(np.uint8)
    elif img.shape[-1] == 3 and img.dtype != np.uint8:
        img = img.astype(np.uint8)
        
    in_h, in_w = int(img.shape[0]), int(img.shape[1])

    # If already correct size, return as is
    if in_w == target_w and in_h == target_h:
        return img

    # Create white canvas
    canvas = np.ones((target_h, target_w, 3), dtype=np.uint8) * 255

    # Resize to fit within target while preserving aspect ratio
    if (in_w > target_w or in_h > target_h) or (in_w < target_w or in_h < target_h):
        if Image is None:
            # Without Pillow, we can only pad/crop when input fits
            if in_w <= target_w and in_h <= target_h:
                # Center pad without resizing
                x0 = (target_w - in_w) // 2
                y0 = (target_h - in_h) // 2
                canvas[y0:y0+in_h, x0:x0+in_w] = img
                return canvas
            raise RuntimeError(
                "Pillow is not installed; cannot resize frames to target window_size. "
                "Install pillow or ensure all PNGs share identical dimensions."
            )

        scale = min(target_w / in_w, target_h / in_h)
        new_w = max(1, int(round(in_w * scale)))
        new_h = max(1, int(round(in_h * scale)))

        # Ensure resized dims are at least 1 and do not exceed target
        new_w = min(new_w, target_w)
        new_h = min(new_h, target_h)

        resized = Image.fromarray(img).resize((new_w, new_h), resample=Image.BICUBIC)
        resized_np = np.asarray(resized, dtype=np.uint8)

        x0 = (target_w - new_w) // 2
        y0 = (target_h - new_h) // 2
        canvas[y0:y0+new_h, x0:x0+new_w] = resized_np

        return canvas


def png_to_mp4(img_dir, output_mp4, fps=20, window_size=(1920, 1080)):
    """
    Convert a directory of PNG images to an MP4 animation.

    Notes:
    - Frames are sorted naturally (frame_0001.png, frame_0002.png, ...).
    - Ensures consistent frame size by resizing and letterboxing to `window_size`.
    - Guarantees even width/height for codec compatibility (yuv420p).
    - Streams frames to the writer to avoid loading all images into memory.
    """

    # Ensure output directory exists
    Path(output_mp4).parent.mkdir(parents=True, exist_ok=True)

    png_paths = sorted(glob.glob(os.path.join(img_dir, "*.png")), key=_natural_key)
    if not png_paths:
        raise FileNotFoundError(f"No PNG files found in {img_dir}")

    # Target frame size
    first = imageio.imread(png_paths[0])
    if window_size is None:
        target_w, target_h = int(first.shape[1]), int(first.shape[0])
    else:
        target_w, target_h = int(window_size[0]), int(window_size[1])

    # Enforce even dimensions for yuv420p
    if target_w % 2 != 0:
        target_w += 1
    if target_h % 2 != 0:
        target_h += 1

    # Stream-encode to MP4
    writer = imageio.get_writer(
        output_mp4,
        fps=fps,
        codec="libx264",
        quality=8,
        pixelformat="yuv420p",
        macro_block_size=1,
    )
    try:
        for fp in tqdm(png_paths):
            img = imageio.imread(fp)
            frame = _prepare_frame(img, target_w, target_h)
            writer.append_data(frame)
    finally:
        writer.close()

    print(f"Animation saved to {output_mp4}")

def animate_residuals(df_sim, df_ml, df_res, sim_name: str, 
                      dir_name: str = 'residuals', fps: int = 20, 
                      window_size: tuple[int, int] = (1920, 1080)):

    img_dir = ROOT / "sims" / sim_name / "animations" / f"{dir_name}"
    img_dir.mkdir(parents=True, exist_ok=True)

    # Save images
    step = df_sim['step'].max()
    start_step = 1
    for s in tqdm(range(start_step, step)):
        fig = plot_compare(df_sim, df_ml, df_res, metric="p", step=s)
        # fig.write_html(f"compare_{s}.html")
        fig.write_image(f"{img_dir}/compare_step{s}.png")
    
    png_to_mp4(
        img_dir=img_dir,
        output_mp4=img_dir.parent / f"{dir_name}-{sim_name}-residual-plots.mp4",
        fps=fps,
        window_size=window_size,  # must be even; will auto-adjust to even
    )   

if __name__ == "__main__":
    # Parameters
    SIM_NAME = "example-model"
    CASE_NAME = 'case0'

    # Generate MP4 animation
    vtu_to_mp4(
        SIM_NAME,
        CASE_NAME, 
        remove_images=True, 
        fps=20, 
        cmap="turbo", 
        off_screen=True, 
        window_size=(1920, 1088)
    )