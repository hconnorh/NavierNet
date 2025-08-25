import os
import subprocess
import configparser

from tqdm import tqdm


class pyfrSimulation:
    """
    A wrapper for running bulk PyFR simulations.
    """
    def __init__(self, sim_name):
        self.sim_name = sim_name
        self.config_dir = os.path.abspath("assets/config") 
        
        self.mesh_file = os.path.join("assets/config", f"2d-cylinder.msh")
        self.pyfrm_file = os.path.join("assets/config", f"2d-cylinder.pyfrm")
        
        self.assets_dir = "assets"
        self.base_dir = "sims"

    def _generate_pyfrm_mesh(self):
        """
        Converts the mesh file (.msh) for the simulation into a PyFR mesh file (.pyfrm)
        using the PyFR import utility. This is required before running the simulation.
        """
        subprocess.run([
            "pyfr", "import", "-t", "gmsh", self.mesh_file, self.pyfrm_file
        ], check=True)
    
    def _modify_ini_file(self, nu, Uin, tend, dt_out, perm_num):
        """
        Generate a customised .ini file in config_dir for a given permutation.
        Uses the base .ini file in assets/config as a template.

        Parameters
        ----------
        nu (float): Kinematic viscosity [m^2/s]
        Uin (float): Inlet velocity [m/s]
        tend (float): End time [s]
        dt_out (float): Output interval [s]
        perm_num (int): Permutation number used to name the output .ini file.

        Returns
        -------
        out_path (str): Path to the newly generated .ini file.
        """

        base_ini = os.path.join(self.assets_dir, "config/2d-cylinder.ini")

        cfg = configparser.RawConfigParser()
        cfg.optionxform = str  # preserve key case
        cfg.read(base_ini)

        if not cfg.has_section("constants"):
            cfg.add_section("constants")
        cfg.set("constants", "nu", str(nu))
        cfg.set("constants", "Uin", str(Uin))
        cfg.set("solver-time-integrator", "tend", str(tend))
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
        self.training_dir = f"{self.sim_dir}/training_data"
        self.sim_config_dir = f"{self.sim_dir}/config"
        for d in [self.sim_dir, self.sim_results_dir, self.animations_dir,
                  self.training_dir, self.sim_config_dir]:
            os.makedirs(d, exist_ok=True)
    
        # Copy Config Files
        os.system(f"cp {self.mesh_file} {self.sim_config_dir}")
        os.system(f"cp {self.pyfrm_file} {self.sim_config_dir}")

        # Copy and Modify .ini file for all permuations
        for perm, (nu, Uin, tend, dt_out) in enumerate(perms):
            ini_path = self._modify_ini_file(nu, Uin, tend, dt_out, perm)
            print(f"Wrote ini: {ini_path}")

    def run(self, pyfrm_file, ini_file, backend=None, results_dir=None, show_progress=True):
        """
        Run a PyFR simulation with the given mesh (.pyfrm) and config (.ini) files.

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
        # GPU or CPU backend
        backend = backend or os.environ.get("PYFR_BACKEND", "openmp")

        # Create results directory if it doesn't exist
        results_dir = results_dir or "results"
        os.makedirs(results_dir, exist_ok=True)
        
        # Requires Absolute Paths #TODO: Not sure why.
        pyfrm_abs = os.path.abspath(pyfrm_file)
        ini_abs = os.path.abspath(ini_file)
        if not os.path.isfile(pyfrm_abs):
            raise FileNotFoundError(f".pyfrm not found: {pyfrm_abs}")
        if not os.path.isfile(ini_abs):
            raise FileNotFoundError(f".ini not found: {ini_abs}")
        
        # Run PyFR simulation as a subprocess
        cmd = ["pyfr"]
        if show_progress:
            cmd.append("-p")
        cmd += [
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
        for c in tqdm(range(len(perms))):
            ini_file = os.path.join(self.sim_config_dir, f"case{c}.ini")
            results_dir = os.path.join(self.sim_results_dir, f"case{c}")
            try:
                self.run(pyfrm_file, ini_file, backend, results_dir, show_progress)
            except Exception as e:
                print(f"Unexpected error running simulation {c}: {e}")
                continue

if __name__ == "__main__":
    sim_name = "2d-cylinder-1s"
    perms = [[
        0.005,  # [nu, m/s^2] Kintematic velocity 
        1,      # [Uin, m/s]  Inlet velocity 
        200,    # [tend, s]   Simulation time
        5       # [dt-out, s] State save delta
    ]]

    m = pyfrSimulation(sim_name)
    m.run_bulk(perms, backend="metal", show_progress=True)