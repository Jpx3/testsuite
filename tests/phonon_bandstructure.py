import ase
import ase.units
import numpy as np
import zipfile
import phonopy
from ase.build.supercells import make_supercell
from ase.calculators.calculator import Calculator
from ase.calculators.emt import EMT
from ase.calculators.lj import LennardJones
from ase.filters import FrechetCellFilter
from ase.optimize import FIRE
from phonopy import Phonopy
from phonopy.harmonic.dynmat_to_fc import get_commensurate_points
from phonopy.phonon.band_structure import get_band_qpoints_by_seekpath, BandStructure
from phonopy.phonon.thermal_properties import ThermalProperties
from phonopy.structure.atoms import PhonopyAtoms
from pymatgen.core import Structure
from pymatgen.io.phonopy import get_phonopy_structure, get_pmg_structure

from driver.driver import Driver

from tqdm import tqdm

THz_to_K = ase.units._hplanck * 1e12 / ase.units._k


def full_phonon_test(
  driver: Driver,
  model_variation: str,
  relax=False
):
  print("Running full phonon test...")
  print("Setting up...")
  import torch
  calc = driver.get_ase_calculator(
    model_variation=model_variation,
    # try to run on cuda if available
    device="cuda" if torch.cuda.is_available() else "cpu"
  )
  print("Calculator: ", calc)

  is_dft = driver.name.startswith("dft")

  # list files in /mdr-phonondb/data
  import os
  data_dir = "tests/mdr-phonondb/data"
  files = os.listdir(data_dir)

  # Generate temp directory
  import tempfile
  import shutil
  temp_dir = tempfile.mkdtemp()
  print(f"Using temporary directory: {temp_dir}")

  results = dict()

  import random
  # Fixed seed for deterministic behavior
  sparse = random.Random(1337)

  for file in tqdm(files):
    if file.endswith(".zip"):
      if sparse.random() < 0.95:
        continue # Process only 5% of files for speed

      with zipfile.ZipFile(os.path.join(data_dir, file), 'r') as zip_ref:
        name = file[:-4]
        file_bytes = zip_ref.read("phonopy_params.yaml.xz")
        # Write bytes to a temporary file
        temp_file_path = os.path.join(temp_dir, f"{name}_phonopy_params.yaml.xz")
        try:
          with open(temp_file_path, 'wb') as temp_file:
            temp_file.write(file_bytes)
          p = phonopy.load(temp_file_path, produce_fc=is_dft)
          if len(p.supercells_with_displacements) < 50:
            results[name] = single_phonon_test(calc, p, relax=relax, just_dft=is_dft, plot_data=False)
        except Exception as e:
          print(f"Error processing {name}: {e}")
        finally:
          # Ensure the temporary file is deleted after use
          if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

  # Remove temp directory
  shutil.rmtree(temp_dir)
  return results


def single_phonon_test(
  calc: Calculator,
  phonon: Phonopy = None,
  relax=False,
  just_dft=False,
  plot_data=False
):
  results = dict()

  if just_dft:
    if relax:
      raise NotImplementedError("DFT relaxation not possible.")
    results["results"] = phonopy_results(phonon)
    return results

  if relax:
    phonon = relax_structure(phonon, calc=calc)

  produce_force_constants(phonon, calc=calc)

  results["results"] = phonopy_results(phonon)
  # if plot_data:
  #   run_band_structure(phonon, plot=True)
  return results


def phonopy_results(
  phonon: phonopy.Phonopy
):
  print("Calculating frequencies...")
  frequencies = get_frequencies(phonon)
  print("Calculating thermal properties...")
  thermo = get_thermal_properties(phonon)
  print("Calculating band structure...")
  band_structure = get_band_structure(phonon)
  print("Calculating DOS...")
  dos = get_dos(phonon)
  return frequencies, thermo, band_structure, dos


# from https://github.com/facebookresearch/fairchem/blob/main/src/fairchem/core/components/calculate/recipes/phonons.py#L57
def relax_structure(
  phonopy_structure: phonopy.Phonopy,
  calc: Calculator,
  fmax: float = 0.0075,
  steps: int = 250,
  **phonopy_kwargs,
):
  primcell = phonopy_atoms_to_ase_atoms(phonopy_structure.primitive)
  primcell.calc = calc
  opt = FIRE(FrechetCellFilter(primcell), logfile=None)
  print("Optimizing primitive cell...")
  opt.run(fmax=fmax, steps=steps)
  print("Optimization done.")
  if phonopy_structure.primitive_matrix is not None:
    P = np.asarray(np.linalg.inv(phonopy_structure.primitive_matrix.T), dtype=np.intc)
    unitcell = make_supercell(primcell, P)
  else:
    unitcell = primcell
  unitcell = ase_to_phonopy_atoms(unitcell)
  supercell_matrix = phonopy_structure.supercell_matrix
  primitive_matrix = phonopy_structure.primitive_matrix
  supercell_matrix = np.ascontiguousarray(supercell_matrix, dtype=int)
  phonon = Phonopy(
    unitcell=unitcell,
    supercell_matrix=supercell_matrix,
    primitive_matrix=primitive_matrix,
    **phonopy_kwargs,
  )
  return phonon


def displace(
  phonon: phonopy.Phonopy,
  distance: float = 0.01,
):
  phonon.generate_displacements(distance=distance)
  return phonon

# Second order taylor expansion of the energy around equilibrium
def produce_force_constants(
  phonon: phonopy.Phonopy,
  calc: Calculator,
):
  # print(len(phonon.supercells_with_displacements), "displaced supercells to calculate forces for.")
  phonon.forces = [
    calc.get_forces(get_pmg_structure(supercell).to_ase_atoms())
    # get_forces(phonopy_atoms_to_ase_atoms(supercell), calc)
    for supercell in tqdm(phonon.supercells_with_displacements)
  ]
  phonon.produce_force_constants()


def get_forces(atoms: ase.Atoms, calc: Calculator) -> np.ndarray:
  atoms.calc = calc
  forces = atoms.get_forces()
  return forces


def get_band_structure(
  phonon: phonopy.Phonopy,
  npoints: int = 100,
) -> BandStructure:
  bands, labels, connections = get_band_qpoints_by_seekpath(
    phonon.primitive, npoints
  )
  phonon.run_band_structure(bands, labels=labels, path_connections=connections)
  return phonon.band_structure


def get_dos(
  phonon: phonopy.Phonopy,
  mesh: tuple[int, int, int] = (20, 20, 20)
):
  phonon.run_mesh(mesh)
  phonon.run_total_dos()
  return phonon.total_dos


def get_frequencies(
  phonon: phonopy.Phonopy,
  qpoints: np.ndarray = None,
):
  if qpoints is None:
    qpoints = get_commensurate_points(phonon.supercell_matrix)
  return np.stack([phonon.get_frequencies(q) for q in tqdm(qpoints)])


def get_thermal_properties(
  phonon: phonopy.Phonopy,
  t_step: float = 10,
  t_min: float = 0,
  t_max: float = 1000,
  mesh: tuple[int, int, int] = (20, 20, 20)
) -> ThermalProperties:
  phonon.run_mesh(mesh)
  phonon.run_thermal_properties(t_step=t_step, t_min=t_min, t_max=t_max)
  return phonon.thermal_properties


def phonopy_atoms_to_ase_atoms(phonopy_atoms: PhonopyAtoms) -> ase.Atoms:
  """Convert a phonopy atoms object to an ASE atoms object."""
  pmg_structure = get_pmg_structure(phonopy_atoms)
  return pmg_structure.to_ase_atoms()


def ase_to_phonopy_atoms(atoms: ase.Atoms) -> PhonopyAtoms:
  if isinstance(atoms, ase.Atoms):
    atoms = Structure.from_ase_atoms(atoms)
  if isinstance(atoms, Structure):
    atoms = get_phonopy_structure(atoms)
  return atoms
