import os
import sys
import subprocess


class pyfrSimulation:
    """
    A wrapper for running bulk PyFR simulations.
    """
    def __init__(self, config_directory, sim_name):
        self.sim_name = sim_name
        self.config_directory = config_directory
        self.mesh_file = os.path.join(self.config_directory, f"{sim_name}.msh")
        self.config_file = os.path.join(self.config_directory, f"{sim_name}.ini")

    def generate_pyfrm_mesh(self):
        "pyfr import -t gmsh /path/to/inc-cylinder.msh /path/to/inc-cylinder.pyfrm"
        self.pyfrm_file = os.path.join(self.config_directory, f"{self.sim_name}.pyfrm")
        subprocess.run([
            "pyfr", "import", "-t", "gmsh", self.mesh_file, self.pyfrm_file
        ], check=True)
    
    def run(self, backend=None):
        backend = backend or os.environ.get("PYFR_BACKEND", "openmp")
        subprocess.run([
            "pyfr", "run", "-b", backend, self.pyfrm_file, self.config_file
        ], check=True)


if __name__ == "__main__":
    config_directory = "ns2d-surrogate/assets/2d-cylinder-config"
    sim_name = "2d-cylinder"

    sim = pyfrSimulation(config_directory, sim_name)
    sim.generate_pyfrm_mesh()
    sim.run(backend="metal" )