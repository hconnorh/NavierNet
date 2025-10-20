import numpy as np
import polars as pl
import pyvista as pv

from typing import Dict, List, Tuple
from tqdm import tqdm
from pathlib import Path

"""
Module: extract_training_data.py
Package: NS2D-Surrogate
Author: @hconnorh
Description:

This module provides utilities to extract nodal pressure and velocity 
time-series from PyFR simulation result files (.vtu). These utilities 
support building ML-ready CSV datasets from time-dependent simulation outputs.

Features:
- Facilitates robust extraction of simulation fields suitable for machine 
  learning workflows.
- Loads VTU files and converts mesh cell data to point data if necessary.
- Extracts nodal pressure and velocity components for each timestep.
- Maps node coordinates to index and manages consistency across different mesh 
  layouts.

Date: 09-May-2023
Modified: 17-Sep-2025
"""
ROOT = Path(__file__).resolve().parent.parent

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


def _build_unique_reference(points_xy: np.ndarray, decimals: int = 12) -> Tuple[np.ndarray, Dict[Tuple[float, float], int]]:
	"""
	Create a unique (x, y) list by rounding coordinates and mapping each rounded
	coordinate to a stable unique node index. Preserves first occurrence order.
	"""
	unique_points: List[Tuple[float, float]] = []
	ref_map: Dict[Tuple[float, float], int] = {}
	for x, y in points_xy:
		k = (float(round(float(x), decimals)), float(round(float(y), decimals)))
		if k not in ref_map:
			ref_map[k] = len(unique_points)
			unique_points.append((float(x), float(y)))
	return np.asarray(unique_points, dtype=float), ref_map


def _aggregate_fields_to_unique(
	ref_map: Dict[Tuple[float, float], int],
	current_xy: np.ndarray,
	p_arr: np.ndarray,
	u_arr: np.ndarray,
	v_arr: np.ndarray,
	decimals: int = 12,
	tol: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	"""
	Aggregate (mean) duplicate points in current_xy onto the unique indices
	defined by ref_map (built from the reference step). Prints a warning if the
	spread across duplicates exceeds tol at any node.

	Point data often comes from per-piece interpolation/averaging of cell data; 
	the neighbor stencil differs across Pieces, so the “same” node gets slightly 
	different averages.
	"""
	N = (max(ref_map.values()) + 1) if ref_map else 0
	acc_p = np.zeros((N,), dtype=float)
	acc_u = np.zeros((N,), dtype=float)
	acc_v = np.zeros((N,), dtype=float)
	acc_c = np.zeros((N,), dtype=np.int64)

	# Track min/max per node for tolerance diagnostics
	min_p = np.full((N,), np.inf)
	max_p = np.full((N,), -np.inf)
	min_u = np.full((N,), np.inf)
	max_u = np.full((N,), -np.inf)
	min_v = np.full((N,), np.inf)
	max_v = np.full((N,), -np.inf)

	for (x, y), pv, uu, vv in zip(current_xy, p_arr, u_arr, v_arr):
		k = (float(round(float(x), decimals)), float(round(float(y), decimals)))
		if k not in ref_map:
			raise KeyError(f"Point not found in reference map for coordinate: {k}")
		i = ref_map[k]
		acc_p[i] += float(pv)
		acc_u[i] += float(uu)
		acc_v[i] += float(vv)
		acc_c[i] += 1
		min_p[i] = pv if pv < min_p[i] else min_p[i]
		max_p[i] = pv if pv > max_p[i] else max_p[i]
		min_u[i] = uu if uu < min_u[i] else min_u[i]
		max_u[i] = uu if uu > max_u[i] else max_u[i]
		min_v[i] = vv if vv < min_v[i] else min_v[i]
		max_v[i] = vv if vv > max_v[i] else max_v[i]

	if np.any(acc_c == 0):
		missing = int(np.count_nonzero(acc_c == 0))
		raise SystemExit(f"Aggregation failure: {missing} reference nodes had no matches in current step")

	p_out = acc_p / acc_c
	u_out = acc_u / acc_c
	v_out = acc_v / acc_c
	vn_out = np.sqrt(u_out * u_out + v_out * v_out)

	# Tolerance check.
	spread_p = max_p - min_p
	spread_u = max_u - min_u
	spread_v = max_v - min_v
	n_exceed = int(np.count_nonzero((spread_p > tol) | (spread_u > tol) | (spread_v > tol)))
	if n_exceed > 0:
		#TODO: Need to check tolerances
		print(f"Warning: {n_exceed} nodes had duplicate values differing by > {tol}:  [p {spread_p.max()}], [u {spread_u.max()}], [v {spread_v.max()}]")

	return p_out, u_out, v_out, vn_out


def extract_csv(sim_name: str, case_name: str, return_df: bool=True) -> None:
	"""
	Extract nodewise results for a single simulation case to CSV.

	Args:
	    sim_name (str): Name of the simulation directory under 'sims/'.
	    case_name (str): Name of the case subdirectory under 'pyfr_results'.
	    return_df (bool): If True, also return the results as a DataFrame.
	"""
	
	# Pathing (use ROOT for robust absolute paths)
	sim_dir = ROOT / "sims" / sim_name
	vtu_dir = sim_dir / "pyfr_results" / case_name
	out_dir = sim_dir / "training_data"
	results = out_dir / f"{case_name}-results.csv"
	key_inputs = out_dir / f"{case_name}-inputs.csv"

	# Find vtu files
	vtus = sorted(vtu_dir.glob("*.vtu"))
	if not vtus:
		raise SystemExit(f"No .vtu files found in {vtu_dir}")

	# Read reference file
	ref_mesh = pv.read(vtus[0])
	ref_mesh = _as_point_data(ref_mesh)
	points = np.asarray(ref_mesh.points)
	if points.shape[1] < 2:
		raise SystemExit("Mesh points must have 2 coordinates")
	ref_xy_full = points[:, :2]
	ref_xy, ref_map = _build_unique_reference(ref_xy_full, decimals=12)
	if ref_xy.shape[0] != ref_xy_full.shape[0]:
		print(f"Deduplicated reference nodes: {ref_xy_full.shape[0]} -> {ref_xy.shape[0]}")

	# Prepare output buffer: rows = num_steps * num_nodes, cols = 7 (step, n_x, n_y, p, u, v, vn)
	num_steps = len(vtus)
	num_nodes = ref_xy.shape[0]

	# Write a stable node map once per case to ensure consistent IDs across datasets
	node_map_path = out_dir / f"{case_name}-node_map.csv"
	node_ids_arr = np.arange(num_nodes, dtype=float)
	node_map_arr = np.column_stack([node_ids_arr, ref_xy[:, 0], ref_xy[:, 1]])
	# Columns: node_id (int-like), n_x, n_y
	np.savetxt(
		node_map_path,
		node_map_arr,
		delimiter=",",
		fmt="%.16g",
		header=",".join(["node_id", "n_x", "n_y"]),
		comments="",
	)

	# Prepare output buffer: rows = num_steps * num_nodes, cols = 8 (step, node_id, n_x, n_y, p, u, v, vn)
	out_arr = np.empty((num_steps * num_nodes, 8), dtype=float)

	# Process each timestep and fill output buffer
	for step, vtu_path in tqdm(list(enumerate(vtus))):
		mesh = pv.read(vtu_path)
		mesh = _as_point_data(mesh)

		xy = np.asarray(mesh.points)[:, :2]

		# Extract fields
		p_arr = _get_pressure(mesh)
		u_arr, v_arr = _get_velocity_components(mesh)
		vnorm_arr = np.sqrt(u_arr * u_arr + v_arr * v_arr)

		# Aggregate duplicates by mean with tolerance diagnostics
		p_arr, u_arr, v_arr, vnorm_arr = _aggregate_fields_to_unique(
			ref_map, xy, p_arr, u_arr, v_arr, decimals=12,
		)

		# Build block for this timestep: [step, node_id, n_x, n_y, p, u, v, vn]
		block = np.column_stack([
			np.full(num_nodes, step, dtype=float),
			node_ids_arr,
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

		# print(f"Processed {step + 1}/{len(vtus)}: {os.path.basename(vtu_path)}")

	# Write once at the end with header
	np.savetxt(
		results,
		out_arr,
		delimiter=",",
		fmt="%.16g",
		header=",".join(["step", "node_id", "n_x", "n_y", "p", "u", "v", "vn"]),
		comments="",
	)

	print(f"Wrote combined CSV: {results}")
	print(f"Wrote node map CSV: {node_map_path}")

	if return_df:
		cols = ["step", "node_id", "n_x", "n_y", "p", "u", "v", "vn"]
		return pl.DataFrame({col: out_arr[:, i] for i, col in enumerate(cols)})

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

	sim_dir = ROOT / "sims" / sim_name
	results_root = sim_dir / "pyfr_results"

	if not results_root.is_dir():
		raise SystemExit(f"{results_root} does not exist or is not a directory")

	# Find all case directories under pyfr_results
	cases = [case_dir for case_dir in sorted(results_root.iterdir()) if case_dir.is_dir()]

	mem_size = 0
	df = None
	for case_dir in cases:
		case_name = case_dir.name
		df = extract_csv(sim_name, case_name, return_df=True)
		# For polars DataFrame, estimate memory usage using estimated_size
		mem_size += df.estimated_size("mb")

	print(f"Results: {mem_size:.2f} MB")
	return df


if __name__ == "__main__":
	sim_name = "water-1200f-3000re"

	# Single Case
	case_name = "case0"
	df = extract_csv(sim_name=sim_name, case_name=case_name) 
	print(f"Results: {df.estimated_size('mb'):.2f} MB")
	print(df.head(5))

	# All cases
	extract_all_cases(sim_name=sim_name) 