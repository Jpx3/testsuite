from collections import defaultdict
from copy import deepcopy

import numpy as np
import torch
from ase import units
from ase.build import molecule
from ase.calculators.calculator import Calculator
from ase.md import VelocityVerlet
from ase.md.npt import NPT
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from tqdm import tqdm

from driver.driver import Driver


def compute_box_length(n_molecules, min_distance=3.5):
  V = n_molecules * min_distance ** 3
  L = V ** (1 / 3)
  return L


def compute_gas_box_length(n_molecules, temperature_K, pressure_bar):
  # N = n_molecules
  # k = Boltzmann constant in eV/K
  # T = temperature in K
  # P = pressure in eV/Å³

  # Calculate the volume per molecule (V/N = kT/P)
  volume_per_molecule_A3 = (units.kB * temperature_K) / (pressure_bar * units.bar)

  # Calculate total volume for N molecules
  total_volume_A3 = n_molecules * volume_per_molecule_A3

  # Get the length of a cubic box
  L = total_volume_A3 ** (1.0 / 3.0)

  print(f"Targeting gas at {temperature_K} K and {pressure_bar} bar.")
  print(f"Volume per molecule: {volume_per_molecule_A3:.1f} Å³")
  print(f"Total gas volume: {total_volume_A3:.1f} Å³")
  print(f"Box length (L): {L:.2f} Å")

  return L

def test_co2_stability(
  driver: Driver,
  model_variation: str = "default"
):
  calc = driver.get_ase_calculator(
    model_variation=model_variation, device=torch.device("cuda"),
    compile=True, syscache=True
  )
  if driver.name == "uma":
    print("UMA driver detected, skipping CO2 stability test.")
    return {}
  print("Calculator obtained:", calc)
  return test_co2_stability_calc(calc)


def test_co2_stability_calc(
  calc: Calculator
):
  np.random.seed(42)

  co2 = molecule('CO2')
  n_molecules = 50
  positions = []
  # L = compute_box_length(n_molecules, min_distance=5)
  L = compute_gas_box_length(n_molecules, temperature_K=300, pressure_bar=1.0)
  min_distance = 3.5  # Minimum distance between molecules
  for i in range(n_molecules):
    offset = np.random.rand(3) * L
    while any(np.linalg.norm(offset - pos) < min_distance for pos in positions):
      offset = np.random.rand(3) * L

    for atom in co2:
      positions.append(atom.position + offset)

  # Flatten all atoms
  symbols = co2.get_chemical_symbols() * n_molecules

  def bond_lengths(atoms):
    co2_bonds = []
    for i in range(0, len(atoms), 3):
      c = atoms[i]
      o1 = atoms[i + 1]
      o2 = atoms[i + 2]
      co2_bonds.append(np.linalg.norm(c.position - o1.position))
      co2_bonds.append(np.linalg.norm(c.position - o2.position))
    return co2_bonds

  results = defaultdict(dict)

  target_pressure = 1.0  # in bar
  temps = np.linspace(300, 2000, 10)
  # run 300K twice
  temps = np.concatenate(([300], temps))

  # Create ASE Atoms object
  from ase import Atoms
  system = Atoms(symbols=symbols, positions=positions)
  system.set_pbc([True, True, True])
  system.set_cell([L, L, L])
  system.calc = calc

  total_steps = 0

  for temp in temps:
    print("Temperature:", temp)
    MaxwellBoltzmannDistribution(system, temperature_K=temp)

    print("Computing initial energy...")
    start_energy = system.get_potential_energy() + system.get_kinetic_energy()

    # dyn = NPT(
    #   system,
    #   timestep=1.0 * units.fs,
    #   externalstress=target_pressure * 1e-4,
    #   temperature_K=temp,
    #   ttime=100 * units.fs,  # temperature coupling time
    #   pfactor=2000 * units.fs  # pressure coupling time
    # )
    dyn = VelocityVerlet(system, timestep=1.0 * units.fs)

    tqdm_iter = tqdm(range(1000))
    for step in tqdm_iter:  # run 1000 steps
      try:
        dyn.run(10)
      except RuntimeError as e:
        # Try again once
        print("Exception during MD step:", e)
        try:
          dyn.run(10)
        except RuntimeError as e2:
          print("Second exception during MD step, skipping to next temperature:", e2)
          break

      # print(f"Step {step}: PE = {system.get_potential_energy():.3f} eV")
      drift = (system.get_potential_energy() + system.get_kinetic_energy()) - start_energy
      # print(f"Energy drift: {drift:.6f} eV")
      # Calculate average C=O bond length
      bonds = bond_lengths(system)
      avg_bond_length = np.mean(bonds)
      # print(f"Average C=O bond length: {avg_bond_length:.3f} Å")
      tqdm_iter.set_description(f"Temp {temp:.1f}K Step {step} Bond {avg_bond_length:.3f}Å Drift {drift:.6f}eV")
      # results[temp][step] = {
      results[total_steps] = {
        "temperature": temp,
        "step": step,
        "volume": system.get_volume(),
        "potential": system.get_potential_energy(),
        "kinetic": system.get_kinetic_energy(),
        "total_energy": system.get_potential_energy() + system.get_kinetic_energy(),
        "drift": drift,
        "positions": deepcopy(system.get_positions()),
        "velocities": deepcopy(system.get_velocities()),
        "forces": deepcopy(system.get_forces()),
        "bonds": bonds,
      }

      total_steps += 1

  return results


if __name__ == "__main__":
  from gpaw import GPAW

  calc = GPAW(mode="lcao", basis="dzp", xc="PBE", txt="gpaw.txt")
  results = test_co2_stability_calc(calc)
  import pickle

  with open("co2_stability_results.pkl", "wb") as f:
    pickle.dump(results, f)
  print("Results saved to co2_stability_results.pkl")
