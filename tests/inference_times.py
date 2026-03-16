import time
from collections import defaultdict

import numpy as np
from ase import Atoms

from data.mptraj import mptraj_testset
from driver.driver import Driver
# from driver.uma_driver import UMADriver

try:
  mptraj = mptraj_testset("data/mptrj-norm-test.db")
except Exception as e:
  print(f"Could not load mptraj test set: {e}")
  # mptraj = mptraj_testset("data/mptrj-norm-test.db")

def generate_atoms(n_atoms: int) -> Atoms:
  """Generate a random Atoms object with n_atoms atoms."""
  size = (n_atoms / 0.033) ** (1/3)
  positions = np.random.rand(n_atoms, 3) * size
  symbols = ["H"] * n_atoms
  cell = np.eye(3) * size
  atoms = Atoms(positions=positions, symbols=symbols, cell=cell, pbc=False)
  return atoms

def test_inference_time(
  driver: Driver,
  model_variant,
  n_atoms_list=None
) -> dict:
  if n_atoms_list is None:
    n_atoms_list = [5, 5, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000, 50000, 100000]
  driver_name = driver.name

  results = defaultdict(dict)

  # run_names = ["gpu"]#, "cpu"]
  run_names = ["cpu"]

  for i in range(1):
    n_atoms = []
    times = []
    name = run_names[i % 2]
    device = "cpu" if name == "cpu" else "cuda"

    calc = driver.get_ase_calculator(model_variant, device=device, compile=True)
    last_time = None

    limit = 10000

    if device == "cpu" and driver.name == "nequix":
      limit = 5000

    for n in n_atoms_list:
      # if i == 0 and n > limit:
      #   # Skip large systems on CPU to save time
      #   print(f"Skipping {n} atoms on CPU to save time")
      #   n_atoms.append(n)
      #   times.append(np.nan)
      #   continue
      needs_a_limit = (driver.name == "uma"
                       or driver.name == "sevennet"
                       or (driver.name == "orb_v3" and model_variant == "conservative_inf_omat")
                       or (driver.name == "nequix")
                       )
      print(f"{driver.name} {needs_a_limit}")

      if n > 1000 and driver.name.lower() == "esen" and device == "cpu":
        # Skip large systems on CPU to save time
        print(f"Skipping {driver.name} inference for {n} atoms on CPU since it would take too much memory ")
        n_atoms.append(n)
        times.append(np.nan)
        continue

      if n > 5000 and device == "cpu" and needs_a_limit:
        # Skip large systems on CPU to save time
        print(f"Skipping {driver.name} inference for {n} atoms on CPU since it would take too much memory ")
        n_atoms.append(n)
        times.append(np.nan)
        continue

      # If not orb and on CPU, skip very large systems
      if n > 10000 and device == "cpu" and driver.name != "orb_v3" and driver.name != "orb_v2":
        # Skip very large systems to save time
        print(f"Skipping {n} atoms to save time")
        n_atoms.append(n)
        times.append(np.nan)
        continue

      n_runs = 20
      for run_idx in range(n_runs):
        # if (n > 5000 and last_time > 120000) or (last_time is not None and last_time > 150000):
        #   print(f"Skipping {n} atoms due to long previous inference time ({last_time:.2f} ms)")
        #   n_atoms.append(n)
        #   times.append(np.nan)
        #   continue

        atoms = generate_atoms(n)
        start_time = time.time_ns()
        atoms.calc = calc
        try:
          atoms.get_forces()
        except Exception as e:
          print(f"Error during force calculation for {n} atoms: {e}")
          n_atoms.append(len(atoms))
          times.append(np.nan)
          break

        end_time = time.time_ns()
        timee = (end_time - start_time) / 1e6
        if run_idx == 0 or n <= 5:
          # Discard the first run to avoid initialization overhead
          print(f"Discarded first run for {n} atoms: {timee:.4f} ms")
          continue
        last_time = timee
        n_atoms.append(len(atoms))
        times.append(timee)
        print(f"{n} atoms: Inference time per step: {timee:.4f} milliseconds")

    results[name]["n_atoms"] = n_atoms
    results[name]["times_ms"] = times

  return {
    "driver": driver_name,
    "model_variant": model_variant,
    "results": results
  }

def test_cachable_inference_time(
  driver: Driver,
  model_variant,
  n_atoms_list=None
) -> dict:
  if n_atoms_list is None:
    n_atoms_list = [10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000, 50000, 100000]
  driver_name = driver.name

  results = defaultdict(dict)

  run_names = ["gpu", "cpu"]

  for i in range(2):
    n_atoms = []
    times = []
    name = run_names[i % 2]
    device = "cpu" if name == "cpu" else "cuda"
    for n in n_atoms_list:
      calc = driver.get_ase_calculator(model_variant, device=device, compile=True, syscache=True)

      needs_a_limit = driver.name == "uma" or driver.name == "sevennet"

      print(f"{driver.name} {needs_a_limit}")


      if n > 1000 and driver.name.lower() == "esen":
        print(f"Skipping {driver.name} inference for {n} atoms for eSEN since it would take too much memory ")
        n_atoms.append(n)
        times.append(np.nan)
        continue

      if n > 5000 and device == "cpu" and needs_a_limit:
        print(f"Skipping {driver.name} inference for {n} atoms on CPU since it would take too much memory ")
        n_atoms.append(n)
        times.append(np.nan)
        continue

      n_runs = 5
      for run_idx in range(n_runs):
        np.random.seed(42)
        atoms = generate_atoms(n)

        start_time = time.time_ns()
        atoms.calc = calc
        try:
          atoms.get_forces()
        except Exception as e:
          print(f"Error during force calculation for {n} atoms: {e}")
          n_atoms.append(len(atoms))
          times.append(np.nan)
          break

        end_time = time.time_ns()
        timee = (end_time - start_time) / 1e6
        if run_idx == 0:
          # Discard the first run to avoid initialization overhead
          print(f"Discarded first run for {n} atoms: {timee:.4f} ms")
          continue
        last_time = timee
        n_atoms.append(len(atoms))
        times.append(timee)
        print(f"{n} atoms: Cachable inference time per step: {timee:.4f} milliseconds")
    results[name]["n_atoms"] = n_atoms
    results[name]["times_ms"] = times
  return {
    "driver": driver_name,
    "model_variant": model_variant,
    "results": results
  }

if __name__ == "__main__":
  exit(1)
  # driver = UMADriver(name="uma")
  # variant = "s-1p1_omat"
  # results = test_inference_time(driver, variant)
  # print(results)