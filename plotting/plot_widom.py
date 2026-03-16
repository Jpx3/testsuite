import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.mplot3d import Axes3D

def plot_pos_approx(folder="../results/mof_co2_insertion/"):
  import os
  per_model_results = {}
  for filename in os.listdir(folder):
    if filename.endswith(".pkl"):
      with open(os.path.join(folder, filename), "rb") as f:
        results = pickle.load(f)
        model_name = filename.split(".pkl")[0]
        per_model_results[model_name] = results

  for model_name, results in per_model_results.items():
    positions, energy_diffs = [], []
    for key, val in results.items():
      if key.startswith("insert_"):
        energy_diffs.append(val["energy_diff_co2_accounted"])
        positions.append(val.get("position", [0, 0, 0]))

    positions = np.array(positions)
    energy_diffs = np.array(energy_diffs)

    # Filter for finite and reasonable energies
    mask = np.isfinite(energy_diffs) & (energy_diffs < 5.0)
    positions, energy_diffs = positions[mask], energy_diffs[mask]

    x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]
    e = energy_diffs

    # --- Filter for the desired XY window ---
    xy_mask = (x >= -6.6) & (x <= 6.6) & (y >= 9) & (y <= 25)
    x, y, z, e = x[xy_mask], y[xy_mask], z[xy_mask], e[xy_mask]

    # --- Create a 3D scatter plot ---
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection='3d')

    sc = ax.scatter(
      x, y, z,
      c=e, cmap='seismic',
      s=12, alpha=0.8,
      vmin=-1, vmax=2
    )

    fig.colorbar(
      sc, ax=ax, label='Energy difference (eV)',
      shrink=0.6, pad=0.1,
      ticks=[-1, 0, 2]
    )

    ax.set_xlabel('X (Å)')
    ax.set_ylabel('Y (Å)')
    ax.set_zlabel('Z (Å)')
    ax.set_title(f'CO$_2$ insertion energy: {model_name}')

    # --- Equal aspect ratio ---
    def set_axes_equal(ax):
      x_limits = ax.get_xlim3d()
      y_limits = ax.get_ylim3d()
      z_limits = ax.get_zlim3d()

      x_range = abs(x_limits[1] - x_limits[0])
      y_range = abs(y_limits[1] - y_limits[0])
      z_range = abs(z_limits[1] - z_limits[0])

      max_range = max([x_range, y_range, z_range])
      mid_x = np.mean(x_limits)
      mid_y = np.mean(y_limits)
      mid_z = np.mean(z_limits)
      ax.set_xlim3d([mid_x - max_range / 2, mid_x + max_range / 2])
      ax.set_ylim3d([mid_y - max_range / 2, mid_y + max_range / 2])
      ax.set_zlim3d([mid_z - max_range / 2, mid_z + max_range / 2])

    set_axes_equal(ax)

    # --- Slightly rotate the camera for a near-top-down perspective ---
    ax.view_init(elev=55, azim=-60)

    plt.tight_layout()
    import os
    if not os.path.exists("figures/co2_insertion"):
      os.makedirs("figures/co2_insertion")
    plt.savefig(f"figures/co2_insertion/co2_insertion_energy_shifted_3d_{model_name}.png", dpi=300)
    plt.show()



def plot_simple_top_down(folder="../results/mof_co2_insertion/"):
  per_model_results = {}
  for filename in os.listdir(folder):
    if filename.endswith(".pkl"):
      with open(os.path.join(folder, filename), "rb") as f:
        results = pickle.load(f)
        model_name = filename.split(".pkl")[0]
        per_model_results[model_name] = results

  for model_name, results in per_model_results.items():
    positions, energy_diffs = [], []
    for key, val in results.items():
      if key.startswith("insert_"):
        energy_diffs.append(val["energy_diff_co2_accounted"])
        positions.append(val.get("position", [0,0,0]))

    positions = np.array(positions)
    energy_diffs = np.array(energy_diffs)

    # Filter energies
    mask = np.isfinite(energy_diffs) & (energy_diffs < 5.0)
    positions, energy_diffs = positions[mask], energy_diffs[mask]

    x, y, z = positions[:,0], positions[:,1], positions[:,2]
    e = energy_diffs
    xy_mask = (x >= -6.6) & (x <= 6.6) & (y >= 9) & (y <= 25)
    x, y, z, e = x[xy_mask], y[xy_mask], z[xy_mask], e[xy_mask]

    # --- Plot ---
    fig = plt.figure(figsize=(8,6))
    fig.gca().set_aspect('equal', adjustable='box')
    ax = fig.add_subplot(111)
    sc = ax.scatter(
      x, y,
      c=e,
      cmap='seismic',
      s=20,
      alpha=0.8,
      vmin=-1,
      vmax=2
    )

    fig.colorbar(
      sc, ax=ax, label='Energy difference (eV)',
      shrink=0.8, pad=0.1,
      ticks=[-1, 0, 2]
    )

    ax.set_xlabel('X (Å)')
    ax.set_ylabel('Y (Å)')
    ax.set_title(f'CO₂ insertion top-down: {model_name}')

    plt.tight_layout()
    if not os.path.exists("figures/co2_insertion"):
      os.makedirs("figures/co2_insertion")
    plt.savefig(f"figures/co2_insertion/co2_insertion_energy_top_down_{model_name}.png", dpi=300)

    plt.show()

def plot_hex_histogram(folder="../results/mof_co2_insertion/"):
  per_model_results = {}
  for filename in os.listdir(folder):
    if filename.endswith(".pkl"):
      with open(os.path.join(folder, filename), "rb") as f:
        results = pickle.load(f)
        model_name = filename.split(".pkl")[0]
        per_model_results[model_name] = results

  for model_name, results in per_model_results.items():
    positions, energy_diffs = [], []
    for key, val in results.items():
      if key.startswith("insert_"):
        energy_diffs.append(val["energy_diff_co2_accounted"])
        positions.append(val.get("position", [0,0,0]))

    positions = np.array(positions)
    energy_diffs = np.array(energy_diffs)

    # Filter energies
    mask = np.isfinite(energy_diffs)# & (energy_diffs < 4.0)
    positions, energy_diffs = positions[mask], energy_diffs[mask]

    eV_to_kJmol = 96.485332123

    x, y, z = positions[:,0], positions[:,1], positions[:,2]
    e = energy_diffs * eV_to_kJmol
    xy_mask = (x >= -6.6) & (x <= 6.6) & (y >= 9) & (y <= 22)
    x, y, z, e = x[xy_mask], y[xy_mask] - 15, z[xy_mask], e[xy_mask]

    def excess_chemical_potential(energy_array, T=300, k_B=1.0):
      """Compute excess chemical potential using the Widom insertion formula."""
      kT = k_B * T
      boltzmann_factors = np.exp(-np.array(energy_array) / kT)
      avg_boltzmann = np.mean(boltzmann_factors)
      mu_excess = -kT * np.log(avg_boltzmann)
      if mu_excess > 100:
        return np.nan
      return mu_excess

    # --- Plot histogram ---
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111)

    hb = ax.hexbin(
      x, y, C=e,
      gridsize=50,
      cmap='seismic',
      reduce_C_function=excess_chemical_potential,
      norm=TwoSlopeNorm(vmin=-60, vcenter=0, vmax=90)
    )

    fig.colorbar(
      hb, ax=ax, label='Free Energy (kJ/mol)',
      shrink=0.8, pad=0.1,
      ticks=np.array([-45, -30, -15, 0])
    )

    ax.set_xlabel('X (Å)')
    ax.set_ylabel('Y (Å)')
    ax.set_title(f'CO₂ Widom analysis for Mg-MOF-74 from {model_name}')

    plt.tight_layout()
    if not os.path.exists("figures/co2_insertion"):
      os.makedirs("figures/co2_insertion")
    plt.savefig(f"figures/co2_insertion/co2_insertion_energy_hexbin_{model_name}.png", dpi=300)
    plt.show()

def plot_honeycomb(folder="../results/mof_co2_insertion/"):
  per_model_results = {}
  for filename in os.listdir(folder):
    if filename.endswith(".pkl"):
      with open(os.path.join(folder, filename), "rb") as f:
        results = pickle.load(f)
        model_name = filename.split(".pkl")[0]
        per_model_results[model_name] = results

  for model_name, results in per_model_results.items():
    positions, energy_diffs = [], []
    for key, val in results.items():
      if key.startswith("insert_"):
        energy_diffs.append(val["energy_diff_co2_accounted"])
        positions.append(val.get("position", [0,0,0]))

    positions = np.array(positions)
    energy_diffs = np.array(energy_diffs)

    # Filter energies
    mask = np.isfinite(energy_diffs) & (energy_diffs < 5.0)
    positions, energy_diffs = positions[mask], energy_diffs[mask]

    # Apply XY mask
    x, y, z = positions[:,0], positions[:,1], positions[:,2]
    e = energy_diffs
    xy_mask = (x >= -6.6) & (x <= 6.6) & (y >= 9) & (y <= 25)
    x, y, z, e = x[xy_mask], y[xy_mask], z[xy_mask], e[xy_mask]

    # --- Generate 2D hex pattern from XY points ---
    # Use actual points as centers of the base hex layer
    hex_points = np.column_stack([x, y])

    # Parameters for honeycomb stacking
    n_layers = 5
    z_spacing = (np.max(z) - np.min(z)) / n_layers if n_layers>0 else 1.0
    a = (np.max(x)-np.min(x))  # approximate hex spacing from your data
    print(f"Hex spacing a: {a:.2f} Å")

    stacked_positions = []
    stacked_energies = []

    for i in range(n_layers):
      z_level = np.min(z) + i * z_spacing
      if i % 2 == 0:
        shift = [0,0]  # even layers
      else:
        shift = [a*3/4, a*np.sqrt(3)/4]  # stagger odd layers

      layer_points = hex_points + shift
      z_col = np.full((layer_points.shape[0],1), z_level)
      stacked_positions.append(np.hstack([layer_points, z_col]))
      stacked_energies.append(e)  # same energies for simplicity

    stacked_positions = np.vstack(stacked_positions)
    stacked_energies = np.hstack(stacked_energies)

    # --- Plot ---
    fig = plt.figure(figsize=(9,7))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(
      stacked_positions[:,0],
      stacked_positions[:,1],
      stacked_positions[:,2],
      c=stacked_energies,
      cmap='seismic',
      s=12,
      alpha=0.8,
      vmin=-1,
      vmax=2
    )

    fig.colorbar(
      sc, ax=ax, label='Energy difference (eV)',
      shrink=0.6, pad=0.1,
      ticks=[np.min(stacked_energies), 0, np.max(stacked_energies)]
    )

    ax.set_xlabel('X (Å)')
    ax.set_ylabel('Y (Å)')
    ax.set_zlabel('Z (Å)')
    ax.set_title(f'CO₂ insertion honeycomb: {model_name}')

    # Equal aspect ratio
    def set_axes_equal(ax):
      x_limits = ax.get_xlim3d()
      y_limits = ax.get_ylim3d()
      z_limits = ax.get_zlim3d()

      x_range = abs(x_limits[1] - x_limits[0])
      y_range = abs(y_limits[1] - y_limits[0])
      z_range = abs(z_limits[1] - z_limits[0])
      max_range = max([x_range, y_range, z_range])

      mid_x = np.mean(x_limits)
      mid_y = np.mean(y_limits)
      mid_z = np.mean(z_limits)
      ax.set_xlim3d([mid_x - max_range/2, mid_x + max_range/2])
      ax.set_ylim3d([mid_y - max_range/2, mid_y + max_range/2])
      ax.set_zlim3d([mid_z - max_range/2, mid_z + max_range/2])

    set_axes_equal(ax)
    ax.view_init(elev=75, azim=-60)
    plt.tight_layout()
    if not os.path.exists("figures/co2_insertion"):
      os.makedirs("figures/co2_insertion")
    plt.savefig(f"figures/co2_insertion/co2_insertion_energy_honeycomb_{model_name}.png", dpi=300)

    plt.show()

if __name__ == "__main__":
  # plot_pos_approx()
  # plot_simple_top_down()
  plot_hex_histogram()
  # plot_honeycomb()
