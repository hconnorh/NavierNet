import subprocess
import glob
import os
from typing import List, Tuple


class PyfrToVtkExporter:
    def __init__(self, mesh_file: str, output_directory: str = "vtk_outputs"):
        self.mesh_file = mesh_file
        self.output_directory = output_directory
        os.makedirs(self.output_directory, exist_ok=True)

        self.base_name = os.path.splitext(os.path.basename(self.mesh_file))[0]
        self.pvd_entries: List[Tuple[float, str]] = []

    def find_solution_files(self) -> List[str]:
        pattern = f"{self.base_name}-*.pyfrs"
        return sorted(glob.glob(pattern))

    def _parse_timestep(self, solution_filename: str) -> float:
        return float(solution_filename.split("-")[-1].split(".")[0])

    def convert(self) -> None:
        pyfrs_files = self.find_solution_files()

        for idx, sol_file in enumerate(pyfrs_files):
            timestep = self._parse_timestep(sol_file)
            out_file = os.path.join(self.output_directory, f"{self.base_name}_{idx:04d}.vtu")

            print(f"Converting {sol_file} -> {out_file}")
            subprocess.run(["pyfr", "export", self.mesh_file, sol_file, out_file])

            self.pvd_entries.append((timestep, os.path.basename(out_file)))

        self._write_pvd()

    def _write_pvd(self) -> None:
        pvd_path = os.path.join(self.output_directory, f"{self.base_name}.pvd")
        with open(pvd_path, "w") as f:
            f.write('<?xml version="1.0"?>\n')
            f.write('<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n')
            f.write('  <Collection>\n')
            for timestep, filename in self.pvd_entries:
                f.write(f'    <DataSet timestep="{timestep}" group="" part="0" file="{filename}"/>\n')
            f.write('  </Collection>\n')
            f.write('</VTKFile>\n')

        print(f"\nDone! PVD file created at: {pvd_path}")
        print("Open this PVD in ParaView to animate your simulation.")


if __name__ == "__main__":
    exporter = PyfrToVtkExporter(mesh_file="inc-cylinder.pyfrm", output_directory="vtk_outputs")
    exporter.convert()
