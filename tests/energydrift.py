import numpy as np
from ase import units
from ase.build import bulk
from ase.md import Langevin

from driver.driver import Driver


def gold_anneahling_test(
  driver: Driver,
  model_variant: str = "default"
):
  calc = driver.get_ase_calculator(model_variant)
  random_state = np.random.RandomState(42)

  TIMESTEP = 2.0  # fs
  FRICTION = 0.02
  SIM_STEPS_PER_TEMP = 5000 # 10 ps at each temperature

  atoms = bulk("Au", "fcc", a=4.08, cubic=True) * (6, 6, 6)
  atoms.calc = calc
  print(f"Created a system with {len(atoms)} atoms.")

  trajectory_data = []

  def export_to_dict(a):
    """
    A function to extract key information from the atoms object
    and append it as a dictionary to our trajectory list.
    """
    frame_data = {
      "time_ps": dyn_nvt.get_time() / units.fs * 1000,
      "temperature_K": a.get_temperature(),
      "positions": a.get_positions(),
      "velocities": a.get_velocities(),
      "cell": a.get_cell(),
      "potential_energy_eV": a.get_potential_energy(),
      "total_energy_eV": a.get_total_energy(),
    }
    trajectory_data.append(frame_data)

  dyn_nvt = Langevin(
    atoms,
    timestep=TIMESTEP,
    temperature_K=300,
    friction=FRICTION
  )

  # Attach our custom function to the dynamics loop
  # It will be called every 100 steps
  dyn_nvt.attach(lambda: export_to_dict(atoms), interval=100)

  # --- 4. The Heating Loop (NVT) ---
  print("\n--- Part 1: Melting Gold (NVT Ensemble) ---")
  temp_initial_K = 300
  temp_final_K = 2000  # Let's run a shorter simulation for demonstration
  temp_step_K = 100

  for temp in range(temp_initial_K, temp_final_K + temp_step_K, temp_step_K):
    print(f"\nSetting temperature to {temp} K")
    dyn_nvt.set_temperature(temp)
    dyn_nvt.run(SIM_STEPS_PER_TEMP)

  print("\nSimulation complete.")
  return trajectory_data