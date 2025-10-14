from process_results import process_sim_results
from extract_training_data import extract_all_cases
from run_sim import pyfrSimulation
from animation import gen_mp4

"""
Module: main.py
Package: NS2D-Surrogate
Author: @hconnorh
Description: 

This script serves as the high-level workflow orchestrator for the NS2D-Surrogate
package. It provides a unified routine to:

    - Run multiple PyFR simulations in batch for given parameter sets
    - Convert simulation results to VTU/VTK format and generate Paraview 
      time-series (.pvd)
    - Extract nodal pressure and velocity time-series data suitable for ML/analysis
    - Optionally generate animations (mp4) for simulation visualisation

The main 'run' function ties together the core modules:
    - run_sim.py:         Handles the setup and launching of PyFR simulations
    - process_results.py: Converts .pyfrs outputs to VTU, generates .pvd files
    - extract_training_data.py: Extracts and merges nodal data into CSVs
    - animation.py:       (Optional) Produces mp4 animations from VTK sequence

Date: 09-May-2023
Modified: 17-Sep-2025
"""

def run(sim_name, cases, save_mp4=False):
    m = pyfrSimulation(sim_name)
    m.run_bulk(cases, backend="metal", show_progress=True)
    process_sim_results(sim_name)
    extract_all_cases(sim_name)
    if save_mp4:
        for i in range(len(cases)):
            gen_mp4(sim_name, f"case{i}")


if __name__ == "__main__":
    sim_name = "test2"
    cases = [
        [
            0.01,   #  [nu] for Re ~ 100
            1.0,    #  [Uin] free-stream velocity (nondim)
            0.05,   #  [dt] solver timestep (adjust if unstable)
            10.0,  #  [tend] total time (long enough for a few vortex shedding cycles)
            0.05    #  [dt_out] output interval
        ], 
    ]

    run(sim_name, cases)

