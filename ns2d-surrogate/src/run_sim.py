import os
import sys
import subprocess


class pyfrSimulation:
    """
    A wrapper for running bulk PyFR simulations.
    """
    def __init__(self, config_directory, sim_name):
        self.sim_name = sim_name
        self.config_directory = os.path.abspath(config_directory)
        self.mesh_file = os.path.join(self.config_directory, f"{sim_name}.msh")
        self.config_file = os.path.join(self.config_directory, f"{sim_name}.ini")
        self.pyfrm_file = os.path.join(self.config_directory, f"{self.sim_name}.pyfrm")

    def generate_pyfrm_mesh(self):
        "pyfr import -t gmsh /path/to/inc-cylinder.msh /path/to/inc-cylinder.pyfrm"
        subprocess.run([
            "pyfr", "import", "-t", "gmsh", self.mesh_file, self.pyfrm_file
        ], check=True)
    
    def run(self, backend=None, results_dir=None, show_progress=True):
        backend = backend or os.environ.get("PYFR_BACKEND", "openmp")
        out_dir = results_dir or os.path.join(self.config_directory, "results")
        os.makedirs(out_dir, exist_ok=True)
        cmd = ["pyfr"]
        if show_progress:
            cmd.append("-p")
        cmd += [
            "run", "-b", backend, self.pyfrm_file, self.config_file
        ]
        subprocess.run(cmd, check=True, cwd=out_dir)


if __name__ == "__main__":
    config_directory = "ns2d-surrogate/assets/2d-cylinder-config"
    sim_name = "2d-cylinder"

    sim = pyfrSimulation(config_directory, sim_name)
    sim.generate_pyfrm_mesh()
    sim.run(backend="metal" )