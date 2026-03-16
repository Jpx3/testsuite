from collections import defaultdict
from copy import deepcopy

import numpy as np
import torch
from ase import Atoms
from ase.build import molecule
from ase.io import read
from tqdm import tqdm

from driver.driver import Driver


def test_co2_insertion_to_mof(
  driver: Driver,
  model_variation: str
):
  calc = driver.get_ase_calculator(
    model_variation=model_variation,
    device=torch.device("cuda")
  )

  # Load MOF-5 structure
  mof = read("tests/MgMOF74.cif")
  # Make PBC explicit
  mof.pbc = [True, True, True]
  mof.calc = calc

  # Create CO2 molecule
  co2 = molecule('CO2')
  co2.cell = np.diag([20.0, 20.0, 20.0])
  co2.calc = calc

  print("Co2 cell:", co2.get_cell())
  mof_energy_without = mof.get_potential_energy()
  co2_energy = co2.get_potential_energy()

  # CO2 energy difference histogram
  results = defaultdict(dict)

  results["metadata"] = {
    "mof_energy_without": mof_energy_without,
    "co2_energy": co2_energy,
    "unit_cell": mof.get_cell().tolist(),
    "mof_positions": mof.get_positions().tolist(),
    "mof_species": mof.get_chemical_symbols(),
    "co2_positions": co2.get_positions().tolist(),
    "co2_species": co2.get_chemical_symbols(),
  }

  n_insertions = 100000

  # Precompute 1000 random rotations
  random_rotations = []
  for _ in range(1000):
    random_rotation = np.random.rand(3, 3)
    u, _, vh = np.linalg.svd(random_rotation)
    random_rotation = u @ vh
    if np.abs(np.linalg.det(random_rotation) + 1) < 1e-6:
      random_rotation[:, 0] = -random_rotation[:, 0]
    if np.abs(np.linalg.det(random_rotation) - 1) > 1e-6:
      raise ValueError("Rotation matrix determinant is not 1 after adjustment: " + str(np.linalg.det(random_rotation)))
    random_rotations.append(random_rotation)

  tqdm_iter = tqdm(range(n_insertions), desc="Widom insertions")
  for idx in tqdm_iter:
    # Randomly position CO2 in the MOF unit cell
    cell = mof.get_cell()
    mof_copy = Atoms(
      symbols=deepcopy(mof.get_chemical_symbols()),
      positions=deepcopy(mof.get_positions()),
      cell=deepcopy(mof.get_cell()),
      pbc=deepcopy(mof.get_pbc()),
      # charges=deepcopy(mof.get_charges()),
      # magmoms=deepcopy(mof.get_magnetic_moments())
    )
    co2_copy = Atoms(
      symbols=deepcopy(co2.get_chemical_symbols()),
      positions=deepcopy(co2.get_positions()),
      cell=deepcopy(co2.get_cell()),
      pbc=deepcopy(co2.get_pbc()),
      # charges=deepcopy(co2.get_charges()),
      # magmoms=deepcopy(co2.get_magnetic_moments())
    )

    random_position = np.array(np.random.rand(3) @ cell)
    while np.linalg.norm(random_position - mof_copy.get_positions(), axis=1).min() < 1:
      random_position = np.array(np.random.rand(3) @ cell)

    random_rotation = random_rotations[np.random.randint(0, 1000)]

    co2_copy.set_positions(co2.get_positions() @ random_rotation + random_position)
    combined = mof_copy + co2_copy
    combined.pbc = [True, True, True]
    combined.calc = calc
    combined_energy = combined.get_potential_energy()
    tqdm_iter.set_postfix({
      "last_energy_diff": combined_energy - mof_energy_without - co2_energy,
      "position": random_position
    })

    results["insert_"+str(idx)] = {
      "position": random_position,
      "combined": combined_energy,
      "energy_diff_co2_accounted": combined_energy - mof_energy_without - co2_energy
    }

  return results


