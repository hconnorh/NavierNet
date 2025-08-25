import os
import glob
import csv
import numpy as np
import pandas as pd
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
	"""If there is no point data but there is cell data, convert"""
	if len(mesh.point_data) == 0 and len(mesh.cell_data) > 0:
		mesh = mesh.cell_data_to_point_data()
	return mesh

def _get_pressure(mesh: "pv.DataSet") -> np.ndarray:
	"""
	Extract the pressure array from the point data of a PyVista mesh.

	Args:
		mesh (pv.DataSet): The mesh containing point data with a 'Pressure' array.

	Returns:
		np.ndarray: The pressure values as a 1D array.
	"""
	PRESSURE_ARRAY_KEY = "Pressure"

	# Check key name
	names = list(mesh.point_data.keys())
	if PRESSURE_ARRAY_KEY not in names:
		raise KeyError(f"Missing pressure key '{PRESSURE_ARRAY_KEY}'. Available: {', '.join(names)}")
	
	# Extract pressure.
	return np.asarray(mesh.point_data[PRESSURE_ARRAY_KEY]).ravel()

def _get_velocity_components(mesh: "pv.DataSet") -> Tuple[np.ndarray, np.ndarray]:
	"""
	Extract the x and y components of the velocity vector from a PyVista mesh.

	Args:
		mesh (pv.DataSet): The mesh containing point data with a 'Velocity' vector.

	Returns:
		Tuple[np.ndarray, np.ndarray]: Arrays of the x and y velocity components.
	"""
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
	"""Use rounding to stabilise float keys"""
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


def extract_csv(sim_name: str, case_name: str, return_df: bool=True) -> None:
	"""
	Extract nodewise results for a single simulation case to CSV.

	Args:
	    sim_name (str): Name of the simulation directory under 'sims/'.
	    case_name (str): Name of the case subdirectory under 'pyfr_results'.
	    return_df (bool): If True, also return the results as a DataFrame.
	"""
	
	# Pathing
	vtu_dir = f"sims/{sim_name}/pyfr_results"
	out_dir = f"sims/{sim_name}/training_data"
	results = f"{out_dir}/{case_name}-results.csv"
	key_inputs = f"{out_dir}/{case_name}-inputs.csv"

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

	# Prepare output buffer: rows = num_steps * num_nodes, cols = 7 (step, n_x, n_y, p, u, v, vn)
	num_steps = len(vtus)
	num_nodes = ref_xy.shape[0]
	out_arr = np.empty((num_steps * num_nodes, 7), dtype=float)

	# Process each timestep and fill output buffer
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

		# Build block for this timestep: [step, n_x, n_y, p, u, v, vn]
		block = np.column_stack([
			np.full(num_nodes, step, dtype=float),
			ref_xy[:, 0],
			ref_xy[:, 1],
			p_arr,
			u_arr,
			v_arr,
			vnorm_arr,
		])
		row_start = step * num_nodes
		row_end = row_start + num_nodes
		out_arr[row_start:row_end, :] = block

		print(f"Processed {step + 1}/{len(vtus)}: {os.path.basename(vtu_path)}")

	# Write once at the end with header
	np.savetxt(
		results,
		out_arr,
		delimiter=",",
		fmt="%.16g",
		header=",".join(["step", "n_x", "n_y", "p", "u", "v", "vn"]),
		comments="",
	)

	print(f"Wrote combined CSV: {results}")

	if return_df:
		cols = ["step", "n_x", "n_y", "p", "u", "v", "vn"]
		return pd.DataFrame(out_arr, columns=cols)

def extract_all_cases(sim_name: str) -> None:
	"""
	Extracts and combines training data CSVs for all simulation cases in a 
	given simulation directory.

	For each case in the simulation, this function calls `extract_csv` to 
	process the results and accumulates the total memory usage of the 
	resulting DataFrames.

	Args:
		sim_name (str): Name of the simulation directory under 'sims/'.
	"""
	# Project Paths
	sim_dir = f"sims/{sim_name}"
	results_root = f"{sim_dir}/pyfr_results"

	# Convert Cases
	cases = ([d for d in sorted(os.listdir(results_root)) if os.path.isdir(os.path.join(results_root, d))])

	mem_size = 0
	for case in cases:
		df = extract_csv(sim_name, case, return_df=True)
		mem_size += df.memory_usage(deep=True).sum() / (1024 ** 2)

	print(f"Results: {mem_size:.2f} MB")


if __name__ == "__main__":
	sim_name = "2d-cylinder-v1"

	# Single Case
	case_name = "case0"
	df = extract_csv(sim_name=sim_name, case_name=case_name) 
	print(f"Results: {df.memory_usage(deep=True).sum() / (1024 ** 2):.2f} MB")
	print(df.head(5))

	# All cases
	extract_all_cases(sim_name=sim_name) 