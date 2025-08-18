# Navier-Stokes 2D ML Surrogate Model

## Training Data

Training data will be captured from PyFR, an open-source CFD package optimsied for GPUs. For our trainging data we will start with modelling incompressible Navier Stokes simulation around a 2D pipe. This is a well known, classic engineering problem.

The data will be on an unstructtured mesh. Data such as velocity, pressure etc, will be captured from the nodes forming raw data for trainging our ML surrogate model.

The objective of the surrogate model will be to use this data to predict the next time step in the model.

## Full Setup

### PyFR Enviroment

```bash
# optional: gmsh (mesh authoring), useful but not required
brew install gmsh

# optional but helpful: libxsmm (speeds OpenMP backend if you use it)
brew install libxsmm

# required for video encoding by imageio / ffmpeg pipeline
brew install ffmpeg
```

Notes:

- On Apple Silicon you will most often use the **Metal** backend (GPU) or the `openmp` backend (CPU). Metal requires `precision = single` in the PyFR config.


### Virtual Enviroment (with UV)

Assumes you already have UV isntalled. If not checkout UV here!

```bash
uv python3 -m venv venv
source venv/bin/activate

# install PyFR + visualization pipeline
uv pip install pyfr pyvista "imageio[ffmpeg]" numpy h5py meshio
```

Packages used in the pipeline:

- `pyfr` — CFD solver
- `pyvista` — VTK/Python rendering (reads `.vtu`)
- `imageio[ffmpeg]` — saves `.mp4` from frames
- `numpy`, `h5py`, `meshio` — common data handling utilities

### Run the simulation (example for Apple M2)

Edit your `inc-cykinder.ini` config so the backend precision suits Metal:

```ini
[backend]
precision = single
```

Run PyFR (Metal GPU backend — recommended for M1/M2):

```bash
pyfr run -b metal mesh/cylinder.msh inc-cylinder.pyfr
```

Outputs: a series of `inc-cylinder-*.pyfrs` solution files

## 4 — Convert solutions to VTK (one-liner concept)

(PyFR provides an exporter; your repo includes a small conversion script — run it.)

```bash
# example conceptual command (the repo includes the script)
python convert_pyfr_to_vtk.py   # creates vtk_outputs/inc-cylinder_XXXX.vtu + inc-cylinder.pvd

```

(You already have this script in the package — no export code required here.)

## 5 — Generate MP4 (Pythonic, no ParaView)

Your package includes a script (e.g. `gen_mp4.py`) that:

1. reads `vtk_outputs/inc-cylinder_*.vtu` with **PyVista**,
2. renders frames off-screen (configurable `window_size`),
3. stitches frames into an MP4 via **imageio**.

Typical usage:

```bash
source venv/bin/activate
python gen_mp4.py
# result: inc-cylinder.mp4 (resolution controlled by `window_size` in the script)

```

Recommended config options to adjust in `gen_mp4.py`:

```python
window_size = (1920, 1080)    # set output resolution (HD)
scalar_name = "Velocity"      # or velocity scalar present in your VTU
fps = 24
cmap = "viridis"
```


## 6 — Troubleshooting (common issues)

- **Missing `libxsmm.dylib` / OpenMP errors:** Homebrew often ships static `.a`. Build `libxsmm` with `make SHARED=1`and place `libxsmm.dylib` in `/opt/homebrew/lib`, then set `DYLD_LIBRARY_PATH`.
- **OpenCL/Metal:** OpenCL is deprecated on macOS; prefer `metal` backend for M1/M2. Metal requires `precision = single`.
- **FFmpeg errors for MP4:** ensure `ffmpeg` is in PATH (`brew install ffmpeg`). For `imageio`, install `imageio[ffmpeg]` in your venv: `pip install "imageio[ffmpeg]"`. If zsh complains about brackets, quote them.
- **Low-resolution video:** increase `window_size` in the PyVista plotter.
- **File naming / time series:** conversion script should produce zero-padded filenames (`inc-cylinder_0000.vtu`, ...) and an `inc-cylinder.pvd` so any reader (if used) recognizes the time series.

## 7 — Minimal Quickstart (copy & paste)

```bash
# 1) Homebrew once
brew install ffmpeg

# 2) venv & pip
python3 -m venv venv
source venv/bin/activate
pip install pyfr pyvista "imageio[ffmpeg]" numpy h5py meshio

# 3) Run PyFR (example for M2 Metal)
pyfr run -b metal mesh/cylinder.msh inc-cylinder.pyfr

# 4) Convert solutions -> vtk (repo includes convert script)
python convert_pyfr_to_vtk.py

# 5) Produce MP4 (repo includes gen_mp4.py)
python gen_mp4.py

```

---

If you want, I can produce a 1-file `quickstart.sh` that performs steps 2–5 (assuming you already installed Homebrew deps). Would you like that?