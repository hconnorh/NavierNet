import os
import glob
import csv
import numpy as np
import pyvista as pv

from typing import Dict, List, Optional, Sequence, Tuple

"""
Utility functions for extracting nodal pressure and velocity timeseries 
generated from pyFR simulation results (.vtu files).

It includes:
- Imports for handling files, arrays, and PyVista meshes.
- Functions to ensure mesh data is in point-data format, extract pressure and 
  velocity components, and build index mappings between point coordinates.
- Designed to robustly handle remapping between different mesh layouts.
"""

def _as_point_data(mesh: "pv.DataSet") -> "pv.DataSet":
	# If there is no point data but there is cell data, convert.
	if len(mesh.point_data) == 0 and len(mesh.cell_data) > 0:
		mesh = mesh.cell_data_to_point_data()
	return mesh

def _get_pressure(mesh: "pv.DataSet") -> np.ndarray:
	PRESSURE_ARRAY_KEY = "Pressure"

	# Check key name
	names = list(mesh.point_data.keys())
	if PRESSURE_ARRAY_KEY not in names:
		raise KeyError(f"Missing pressure key '{PRESSURE_ARRAY_KEY}'. Available: {', '.join(names)}")
	
	# Extract pressure.
	return np.asarray(mesh.point_data[PRESSURE_ARRAY_KEY]).ravel()

def _get_velocity_components(mesh: "pv.DataSet") -> Tuple[np.ndarray, np.ndarray]:
	VELOCITY_VECTOR_KEY = "Velocity"

	# Check key name
	names = list(mesh.point_data.keys())
	if VELOCITY_VECTOR_KEY not in names:
		raise KeyError(f"Missing velocity vector key '{VELOCITY_VECTOR_KEY}'. Available: {', '.join(names)}")

	# Check dimensions
	vec = np.asarray(mesh.point_data[VELOCITY_VECTOR_KEY])
	if vec.ndim != 2 or vec.shape[1] < 2:
		raise ValueError(f"Velocity vector must be 2D/3D, got {vec.shape}")

	# Return x and y components
	return vec[:, 0].ravel(), vec[:, 1].ravel()


def _build_reference_index(points_xy: np.ndarray) -> Dict[Tuple[float, float], int]:
	# Use rounding to stabilize float keys
	keys = [(
		float(round(x, 12)),
		float(round(y, 12))
	) for x, y in points_xy]
	return {k: i for i, k in enumerate(keys)}


def _compute_remap_index(reference_xy: np.ndarray, current_xy: np.ndarray) -> np.ndarray:
	"""
	Return indices idx such that reference[i] == current[idx[i]] by coordinates.
	If layouts are identical, returns np.arange(N).
	"""
	if reference_xy.shape != current_xy.shape:
		raise ValueError("Mesh point count changed between timesteps.")

	if np.allclose(reference_xy, current_xy, rtol=0.0, atol=0.0):
		return np.arange(reference_xy.shape[0])

	ref_map = _build_reference_index(reference_xy)
	idx: List[int] = []
	for x, y in current_xy:
		k = (float(round(x, 12)), float(round(y, 12)))
		if k not in ref_map:
			raise KeyError("Point not found in reference mesh for coordinate: %r" % (k,))
		idx.append(ref_map[k])
	return np.asarray(idx, dtype=np.int64)


def extract_csv(sim_name: str, case_name:str) -> None:
	"""
	Extract csv for a single case.
	"""
	
	# Pathing
	vtu_dir = f"sims/{sim_name}/pyfr_results"
	out_dir = f"sims/{sim_name}/training_data"
	os.makedirs(f"{out_dir}/{case_name}", exist_ok=True)
	combined_file = f"{out_dir}/{case_name}/all.csv"

	# Find vtu files
	vtu_dir = f"{vtu_dir}/{case_name}"
	vtus = sorted(glob.glob(os.path.join(vtu_dir, "*.vtu")))
	if not vtus:
		raise SystemExit(f"No .vtu files found in {vtu_dir}")

	# Read reference file
	ref_mesh = pv.read(vtus[0])
	ref_mesh = _as_point_data(ref_mesh)
	points = np.asarray(ref_mesh.points)
	if points.shape[1] < 2:
		raise SystemExit("Mesh points must have 2 coordinates")
	ref_xy = points[:, :2]

	# Prepare combined CSV writer with header
	all_f = open(combined_file, "w", newline="")
	all_writer = csv.writer(all_f)
	all_writer.writerow(["n_x", "n_y", "p", "u", "v", "vn"])

	# Process each timestep and append rows (one per node)

	for step, vtu_path in enumerate(vtus):
		mesh = pv.read(vtu_path)
		mesh = _as_point_data(mesh)

		xy = np.asarray(mesh.points)[:, :2]
		if np.allclose(xy, ref_xy, rtol=0.0, atol=0.0):
			idx = None
		else:
			idx = _compute_remap_index(ref_xy, xy)

		# Extract fields
		p_arr = _get_pressure(mesh)
		u_arr, v_arr = _get_velocity_components(mesh)
		vnorm_arr = np.sqrt(u_arr * u_arr + v_arr * v_arr)

		if idx is not None:
			p_arr = p_arr[idx]
			u_arr = u_arr[idx]
			v_arr = v_arr[idx]
			vnorm_arr = vnorm_arr[idx]

		# Write one row per node for this timestep
		for i in range(ref_xy.shape[0]):
			all_writer.writerow([
				f"{ref_xy[i, 0]:.16g}",
				f"{ref_xy[i, 1]:.16g}",
				f"{p_arr[i]:.16g}",
				f"{u_arr[i]:.16g}",
				f"{v_arr[i]:.16g}",
				f"{vnorm_arr[i]:.16g}",
			])

		print(f"Processed {step + 1}/{len(vtus)}: {os.path.basename(vtu_path)}")

	# Close file
	all_f.close()
	print(f"Wrote combined CSV: {combined_file}")


if __name__ == "__main__":
	
	sim_name = "2d-cylinder-v1"
	case_name = "case0"
	extract_csv(sim_name=sim_name, case_name=case_name) 
