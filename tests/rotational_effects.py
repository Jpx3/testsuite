import numpy as np
import torch
from ase import Atoms
from tqdm import tqdm

from data.mptraj import mptraj_testset
from driver.driver import Driver

try:
  mptraj = mptraj_testset("data/mptrj-norm-test.db")
except Exception as e:
  print(f"Could not load mptraj test set: {e}")
  # mptraj = mptraj_testset("data/mptrj-norm-test.db")

def test_rotational_effects(
  driver: Driver,
  model_variation: str
):
  device = torch.device("cuda")
  calc = driver.get_ase_calculator(
    model_variation=model_variation,
    device=device
  )

  # First 1000 elements of the mptraj list

  tqdm_iter = tqdm(mptraj, desc="Testing rotational effects", total=1000)

  n = 1000
  x = 0

  results = {}

  for ase_atoms in tqdm_iter:
    ase_atoms: Atoms = ase_atoms
    n_a = 8
    n_b = 8

    # Print atom and atom info data
    # print(f"Testing structure {x} with {len(ase_atoms)} atoms.")
    # print(f"Info: {ase_atoms.info}")

    initial_energy = ase_atoms.get_potential_energy()
    initial_forces = ase_atoms.get_forces()

    # print(f"Initial energy: {initial_energy}")
    # print(f"Initial forces: {initial_forces}")

    n = n - 1
    if n <= 0:
      break

    linspace_a = torch.linspace(0, 2 * torch.pi, steps=n_a).cpu()
    linspace_b = torch.linspace(0, 2 * torch.pi, steps=n_b).cpu()

    energies = torch.zeros((n_a, n_b), device=device)
    forces = torch.zeros((n_a, n_b, len(ase_atoms), 3), device=device)

    positions = ase_atoms.get_positions()
    cell = ase_atoms.get_cell()

    for i, angle_a in enumerate(linspace_a):
      for j, angle_b in enumerate(linspace_b):
        rotation_matrix_a = np.array([
          [np.cos(angle_a), -np.sin(angle_a), 0],
          [np.sin(angle_a), np.cos(angle_a), 0],
          [0, 0, 1]
        ])
        rotation_matrix_b = np.array([
          [1, 0, 0],
          [0, np.cos(angle_b), -np.sin(angle_b)],
          [0, np.sin(angle_b), np.cos(angle_b)]
        ])
        rotation_matrix = rotation_matrix_b @ rotation_matrix_a
        rotated_positions = positions @ rotation_matrix.T
        rotated_cell = cell @ rotation_matrix.T
        ase_atoms.set_positions(rotated_positions)
        ase_atoms.set_cell(rotated_cell, scale_atoms=False)
        # ase_atoms.set_pbc([True, True, True])
        ase_atoms.calc = calc
        try:
          energy = ase_atoms.get_potential_energy()
          force = ase_atoms.get_forces()
          energies[i, j] = torch.tensor(energy, device=device)
          forces[i, j] = torch.tensor(force, device=device)
        except Exception as e:
          print(f"Error computing energy/forces for structure {x} at angles ({angle_a}, {angle_b}): {e}")
          energies[i, j] = torch.tensor(float('nan'), device=device)
          forces[i, j] = torch.tensor(float('nan'), device=device)
    # Save everything to results
    results[f'structure_{x}'] = {
      'energy_initial': initial_energy,
      'forces_initial': initial_forces,
      'energy_predicted': energies.cpu().numpy(),
      'forces_predicted': forces.cpu().numpy(),
      'positions': torch.tensor(positions, device='cpu').numpy(),
      'cell': torch.tensor(cell, device='cpu').numpy()
    }
    x += 1
  return results


