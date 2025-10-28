import os
import subprocess
import configparser
import polars as pl
from pathlib import Path
import shutil
import sys

from tqdm import tqdm


"""
Module: run_sim.py
Package: NavierNet
Author: @hconnorh
Description: 

This module provides the pyfrSimulation class, a utility for running PyFR 
simulations, preparing and customising input files and managing simulation 
assets and configuration. It is used in automated workflows for parameter 
studies or ML training data generation.
"""


ROOT = Path(__file__).resolve().parent.parent

class PyfrSimulation:
    """
    A wrapper for running bulk PyFR simulations.
    """
    def __init__(self, sim_name, mesh_file=None, 
                       pyfrm_file=None, ini_file=None):
        self.sim_name = sim_name

        # Resolve paths relative to repo root 
        assets_config_dir = ROOT / "assets" / "config"

        # Default files if not provided
        if mesh_file is None:
            mesh_file = assets_config_dir / "2d-cylinder.msh"
        if pyfrm_file is None:
            pyfrm_file = assets_config_dir / "2d-cylinder.pyfrm"
        if ini_file is None:
            ini_file = assets_config_dir / "2d-cylinder.ini"

        # Store absolute paths as strings for downstream use
        self.mesh_file = str(Path(mesh_file).resolve())
        self.pyfrm_file = str(Path(pyfrm_file).resolve())
        self.ini_file = str(Path(ini_file).resolve())

        self.assets_dir = str((ROOT / "assets").resolve())
        self.base_dir = str((ROOT / "sims").resolve())
        self.config_dir = str(assets_config_dir.resolve()) 

    def _generate_pyfrm_mesh(self):
        """
        Converts the mesh file (.msh) for the simulation into a PyFR mesh file 
        (.pyfrm) using the PyFR import utility. This is required before running 
        the simulation.
        """
        cmd = self._pyfr_base_cmd(show_progress=False) + [
            "import", "-t", "gmsh", self.mesh_file, self.pyfrm_file
        ]
        try:
            subprocess.run(cmd, check=True)
        except FileNotFoundError as e:
            raise FileNotFoundError(
                "PyFR CLI not found. Ensure PyFR is installed and on PATH, or "
                "install into this env"
            ) from e

    def _pyfr_base_cmd(self, show_progress):
        """
        Return the base PyFR command, preferring the CLI if available, 
        otherwise python -m pyfr.
        """
        pyfr_exe = shutil.which("pyfr")
        cmd = [pyfr_exe] if pyfr_exe else [sys.executable, "-m", "pyfr"]
        if show_progress:
            cmd.append("-p")
        return cmd
    
    @staticmethod
    def _running_in_notebook():
        """Best-effort detection of Jupyter/IPython notebook environment."""
        try:
            from IPython import get_ipython  # type: ignore
            ip = get_ipython()
            if ip and getattr(ip, 'kernel', None) is not None:
                return True
        except Exception:
            return False
        # Fallback checks
        return 'ipykernel' in sys.modules or os.environ.get('JPY_PARENT_PID') is not None
    
    def _modify_ini_file(self, nu, Uin, dt, tend, dt_out, perm_num):
        """
        Generate a customised .ini file in config_dir for a given permutation.
        Uses the base .ini file in assets/config as a template.

        Parameters
        ----------
        nu (float): Kinematic viscosity [m^2/s]
        Uin (float): Inlet velocity [m/s]
        dt (float): Time step
        tend (float): End time [s]
        dt_out (float): Output interval [s]
        perm_num (int): Permutation number used to name the output .ini file.

        Returns
        -------
        out_path (str): Path to the newly generated .ini file.
        """
        cfg = configparser.RawConfigParser()
        cfg.optionxform = str  # preserve key case
        ini_path = str(Path(self.ini_file).resolve())
        read_ok = cfg.read(ini_path)

        if not read_ok:
            raise FileNotFoundError(f"Base .ini template not found: {ini_path}")

        if not cfg.has_section("constants"):
            cfg.add_section("constants")
        cfg.set("constants", "nu", str(nu))
        cfg.set("constants", "Uin", str(Uin))

        if not cfg.has_section("solver-time-integrator"):
            cfg.add_section("solver-time-integrator")
        cfg.set("solver-time-integrator", "dt", str(dt))
        cfg.set("solver-time-integrator", "tend", str(tend))

        if not cfg.has_section("soln-plugin-writer"):
            cfg.add_section("soln-plugin-writer")
        cfg.set("soln-plugin-writer", "dt-out", str(dt_out))

        out_name = f"case{perm_num}.ini"
        out_path = os.path.join(self.sim_config_dir, out_name)
        with open(out_path, "w") as f:
            cfg.write(f)

        return out_path

    def _new_sim_project(self, perms):
        """
        Creates a new simulation project directory structure and prepares 
        configuration files for all specified parameter permutations.

        Parameters
        ----------
        perms (list): List of parameter tuples, where each tuple contains 
                      (nu, Uin, tend, dt_out) for a simulation case.
        """
        
        # Make project directory structure
        self.sim_dir = f"{self.base_dir}/{self.sim_name}"
        self.sim_results_dir = f"{self.sim_dir}/pyfr_results"
        self.animations_dir = f"{self.sim_dir}/animations"
        self.training_dir = f"{self.sim_dir}/ml_training"
        self.sim_config_dir = f"{self.sim_dir}/config"
        for d in [self.sim_dir, self.sim_results_dir, self.animations_dir,
                  self.training_dir, self.sim_config_dir]:
            os.makedirs(d, exist_ok=True)
    
        # Copy Config Files
        shutil.copy(self.mesh_file, self.sim_config_dir)
        shutil.copy(self.pyfrm_file, self.sim_config_dir)

        # Copy and Modify .ini file for all permutations
        case_params = []
        for perm, (nu, Uin, dt, tend, dt_out) in enumerate(perms):
            ini_path = self._modify_ini_file(nu, Uin, dt, tend, dt_out, perm)
            print(f"New ini file: {ini_path}")

            # Convert the dictionary to a DataFrame and save as CSV
            case_params.append({"case": perm, "nu" : nu, "Uin" : Uin, 
                                "tend": tend,"dt_out":dt_out})
       
        df = pl.DataFrame(case_params)
        df.write_csv(f"{self.training_dir}/case-inputs.csv")


    def run(self, pyfrm_file, ini_file, backend=None, 
                  results_dir=None, show_progress=True):
        """
        Run a PyFR simulation with the given mesh (.pyfrm) and configuration
        from (.ini) files.

        Parameters
        ----------
        pyfrm_file (str): Path to the mesh file.
        ini_file (str): Path to the config file.
        backend (str): Compute backend to use (default: 'openmp' or PYFR_BACKEND env).
        results_dir (str): Directory to store results (default: 'results').
        show_progress (bool): Show PyFR progress output (default: True).

        Raises
        ------
        FileNotFoundError: If input files are missing.
        subprocess.CalledProcessError: If the simulation fails.
        """
        # GPU or CPU backend; default to Metal on Apple Silicon
        backend = backend or os.environ.get("PYFR_BACKEND", "metal")

        # Create results directory if it doesn't exist
        results_dir = results_dir or "results"
        os.makedirs(results_dir, exist_ok=True)
        
        pyfrm_abs = os.path.abspath(pyfrm_file)
        ini_abs = os.path.abspath(ini_file)
        if not os.path.isfile(pyfrm_abs):
            raise FileNotFoundError(f".pyfrm not found: {pyfrm_abs}")
        if not os.path.isfile(ini_abs):
            raise FileNotFoundError(f".ini not found: {ini_abs}")
        
        # Run PyFR simulation as a subprocess
        # PyFR's '-p' progress output is not notebook-friendly; disable in notebooks
        effective_progress = bool(show_progress) and not self._running_in_notebook()
        cmd = self._pyfr_base_cmd(show_progress=effective_progress) + [
            "run", "-b", backend, pyfrm_abs, ini_abs
        ]
        subprocess.run(cmd, check=True, cwd=results_dir)


    def run_bulk(self, perms, backend=None, mesh_file="2d-cylinder.pyfrm", show_progress=False):
        """
        Run simulations for all parameter permutations.

        Parameters
        ----------
        perms (list): List of parameter sets for each simulation.
        backend (str): Compute backend to use (default: None).
        mesh_name (str): Name of the mesh file.
        """

        # Setup new project
        self._new_sim_project(perms)
        pyfrm_file = os.path.join(self.sim_config_dir, mesh_file)

        # Run all cases into separate result subdirs
        print(f"Running {len(perms)} simulations...")
        pbar = tqdm(range(len(perms))) if not show_progress else range(len(perms))
        for c in pbar:
            ini_file = os.path.join(self.sim_config_dir, f"case{c}.ini")
            results_dir = os.path.join(self.sim_results_dir, f"case{c}")
            try:
                self.run(pyfrm_file, ini_file, backend, results_dir, show_progress)
            except Exception as e:
                print(f"Unexpected error running simulation {c}: {e}")
                continue

if __name__ == "__main__":
    sim_name = "example-model4"
    perms = [
        [
            0.01,       # [nu, m/s^2] Kinematic velocity 
            2.0,        # [Uin, m/s]  Inlet velocity 
            0.05,       # [dt, s]     Time step
            60.0,       # [tend, s]   Simulation total time
            0.05,       # [dt-out, s] Save state # Sh
        ],
    ]
    m = PyfrSimulation(sim_name)
    m.run_bulk(perms, backend="metal", show_progress=True)