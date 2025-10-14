import os
import sys
import glob
import shutil
import subprocess
import re

from typing import List, Tuple

"""
Module: process_results.py
Package: NS2D-Surrogate
Author: @hconnorh
Description: 

This file provides utility functions for handling simulation data files. It 
includes:
	- Finding mesh files (.pyfrm) in a simulation's config directory.
	- Writing .pvd files to describe time series of VTU files.
	- Converting simulation result files (.pyfrs) to .VTU format and generating 
      extracting nodal results.

Date: 09-May-2023
Modified: 17-Sep-2025
"""

def find_mesh_file(sim_dir: str) -> str:
    """
    Find the first .pyfrm mesh file in the config directory of the simulation.

    Args:
        sim_dir (str): Path to the simulation directory.
    """
    cfg_dir = os.path.join(sim_dir, "config")
    pyfrm_files = sorted(glob.glob(os.path.join(cfg_dir, "*.pyfrm")))
    if not pyfrm_files:
        raise SystemExit(f"No .pyfrm mesh found under {cfg_dir}")
    if len(pyfrm_files) > 1:
        print("WARNING: more than one mesh found. First taken")
    return pyfrm_files[0]

def write_pvd(pvd_entries: List[Tuple[float, str]], out_dir: str, base_name: str) -> None:
    """
    Write a .pvd file listing VTU files and their timesteps for ParaView time series visualization.

    Args:
        pvd_entries (List[Tuple[float, str]]): List of (timestep, filename) pairs.
        out_dir (str): Output directory for the .pvd file.
        base_name (str): Base name for the .pvd file.
    """
    pvd_path = os.path.join(out_dir, f"{base_name}.pvd")
    with open(pvd_path, "w") as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n')
        f.write('  <Collection>\n')
        for timestep, filename in pvd_entries:
            f.write(f'    <DataSet timestep="{timestep}" group="" part="0" file="{filename}"/>\n')
        f.write('  </Collection>\n')
        f.write('</VTKFile>\n')
    print(f"PVD created at: {pvd_path}")

def convert_case(sim_dir: str, case: str) -> None:
    """
    Convert all .pyfrs files in a case directory to .vtu files and generate a PVD file.

    Args:
        sim_dir (str): Path to the simulation directory.
        case (str): Name of the case subdirectory under pyfr_results.
    """
    case_dir = f"{sim_dir}/pyfr_results/{case}"
    mesh_file = find_mesh_file(sim_dir)
    out_dir = case_dir

    # Discover series files (unsorted; we'll sort numerically by extracted time)
    pyfrs_files = list(glob.glob(os.path.join(case_dir, "*.pyfrs")))
    if not pyfrs_files:
        print(f"No .pyfrs found in {case_dir}; skipping")
        return None

    # Derive prefix from first file: e.g. file-0.00.pyfrs -> file-0000.vtu
    first_stem = os.path.splitext(os.path.basename(pyfrs_files[0]))[0]
    prefix = first_stem.rsplit("-", 1)[0] if "-" in first_stem else first_stem
    pvd_entries: List[Tuple[float, str]] = []

    # Extract numeric timesteps from filenames using regex; handle values like 0.00, 2.00, 19.95
    time_pattern = re.compile(r"-(?P<time>[0-9]+(?:\.[0-9]+)?)\.pyfrs$")
    indexed: List[Tuple[float, str]] = []
    for f in pyfrs_files:
        name = os.path.basename(f)
        m = time_pattern.search(name)
        if not m:
            # Skip files that don't match expected pattern
            continue
        t = float(m.group("time"))
        indexed.append((t, f))

    if not indexed:
        print(f"No .pyfrs with parseable time suffix found in {case_dir}; skipping")
        return None

    # Sort by numeric time, then by filename for stability
    indexed.sort(key=lambda x: (x[0], x[1]))

    # Decide PyFR invocation
    pyfr_exe = shutil.which("pyfr")
    if pyfr_exe:
        cmd_base = [pyfr_exe]
    else:
        cmd_base = [sys.executable, "-m", "pyfr"] # Fallback to module

    # Convert files in numeric time order and assign sequential indices
    for i, (t, sol_abspath) in enumerate(indexed):
        sol_name = os.path.basename(sol_abspath)
        out_file_rel = f"{prefix}_{i:04d}.vtu"
        
        # Convert file
        print(f"Converting {sol_name} -> {os.path.join(out_dir, os.path.basename(out_file_rel))}")
        mesh_rel_to_case = os.path.relpath(mesh_file, start=case_dir)
        subprocess.run(cmd_base + ["export", mesh_rel_to_case, sol_name,
                                   out_file_rel], check=True, cwd=case_dir)

		# Append pvd file using parsed numeric time
        pvd_entries.append((t, os.path.basename(out_file_rel)))
	
    write_pvd(pvd_entries, out_dir, prefix)

def process_sim_results(sim_name: str) -> None:
    """
    Convert all PyFR case results for a given simulation into VTK format and generate PVD files.

    Args:
        sim_name (str): Name of the simulation directory under 'sims/'.
    """
    # Project Paths
    sim_dir = f"sims/{sim_name}"
    results_root = f"{sim_dir}/pyfr_results"
	
    # Convert Cases
    cases = [d for d in sorted(os.listdir(results_root)) 
             if os.path.isdir(os.path.join(results_root, d))]
    for case in cases:
        convert_case(sim_dir, case)

if __name__ == "__main__":

    # Process pyfrs results
    sim_name = "sim-3600f"
    process_sim_results(sim_name)
	