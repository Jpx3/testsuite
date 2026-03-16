import pickle
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm
from collections import defaultdict
from scipy.stats import linregress

if __name__ == "__main__":
  folder = "../results/co2_stability/"
  import os

  files = os.listdir(folder)

  temperatures = defaultdict(dict)
  steps = defaultdict(dict)
  volumes = defaultdict(dict)
  potential_energies = defaultdict(dict)
  kinetic_energies = defaultdict(dict)
  total_energies = defaultdict(dict)
  drifts = defaultdict(dict)
  positions = defaultdict(dict)
  velocities = defaultdict(dict)
  forces = defaultdict(dict)
  bonds = defaultdict(dict)

  for file in tqdm(files):
    if file.endswith(".pkl"):
      model_name = file.split(".")[0]
      dat = pickle.load(open(os.path.join(folder, file), "rb"))

      if model_name == "painn_oc20":
        continue

      for frame_idx in dat:
        if frame_idx < 1000:
          continue

        simulation_frame = dat[frame_idx]
        temperatures[model_name][frame_idx] = simulation_frame["temperature"]
        steps[model_name][frame_idx] = frame_idx
        volumes[model_name][frame_idx] = simulation_frame["volume"]
        potential_energies[model_name][frame_idx] = simulation_frame["potential"]
        kinetic_energies[model_name][frame_idx] = simulation_frame["kinetic"]
        total_energies[model_name][frame_idx] = simulation_frame["total_energy"]
        drifts[model_name][frame_idx] = simulation_frame["drift"]
        positions[model_name][frame_idx] = simulation_frame["positions"]
        velocities[model_name][frame_idx] = simulation_frame["velocities"]
        forces[model_name][frame_idx] = simulation_frame["forces"]
        bonds[model_name][frame_idx] = simulation_frame["bonds"]

  # Shift all frame indices to start from zero
  for model_name in steps:
    min_frame = min(steps[model_name].keys())
    steps[model_name] = {k - min_frame: v - min_frame for k, v in steps[model_name].items()}
    temperatures[model_name] = {k - min_frame: v for k, v in temperatures[model_name].items()}
    volumes[model_name] = {k - min_frame: v for k, v in volumes[model_name].items()}
    potential_energies[model_name] = {k - min_frame: v for k, v in potential_energies[model_name].items()}
    kinetic_energies[model_name] = {k - min_frame: v for k, v in kinetic_energies[model_name].items()}
    total_energies[model_name] = {k - min_frame: v for k, v in total_energies[model_name].items()}
    drifts[model_name] = {k - min_frame: v for k, v in drifts[model_name].items()}
    positions[model_name] = {k - min_frame: v for k, v in positions[model_name].items()}
    velocities[model_name] = {k - min_frame: v for k, v in velocities[model_name].items()}
    forces[model_name] = {k - min_frame: v for k, v in forces[model_name].items()}
    bonds[model_name] = {k - min_frame: v for k, v in bonds[model_name].items()}

  # Create folder for plots
  plot_folder = "figures/co2_stability/"
  os.makedirs(plot_folder, exist_ok=True)

  from model_colors import model_colors

  # Total Energy vs Step Plot

  plt.figure(figsize=(8, 6))
  for model_name in steps:
    step_values = sorted(steps[model_name].keys())
    energy_values = [total_energies[model_name][step] for step in step_values]
    color = model_colors.get(model_name, None)
    time_ps = np.array(step_values) * 0.01  # assuming 10 fs per step
    plt.plot(time_ps, energy_values, label=model_name, color=color)
  plt.xlabel("Time (ps)")
  plt.ylabel("Total Energy (eV)")
  plt.title("Total Energy vs Step for CO$_2$ Simulations")
  plt.legend()
  plt.grid()
  plt.savefig(os.path.join(plot_folder, "total_energy_vs_step.png"), dpi=400)
  plt.show()

  # Compute energy drift per ps at each temp
  print("Energy Drift per ps:")
  for model_name in drifts:
    drift_values = np.array([drifts[model_name][step] for step in sorted(drifts[model_name].keys())])
    time_ps = np.array(sorted(drifts[model_name].keys())) * 0.01
    avg_drift_per_ps = np.mean(drift_values) / (time_ps[-1] - time_ps[0])
    print(f"{model_name}: {avg_drift_per_ps:.4e} eV/ps")


  # Potential Energy vs Step Plot
  plt.figure(figsize=(8, 6))
  for model_name in steps:
    step_values = sorted(steps[model_name].keys())
    energy_values = [potential_energies[model_name][step] for step in step_values]
    color = model_colors.get(model_name, None)
    time_ps = np.array(step_values) * 0.01  # assuming 10 fs per step
    plt.plot(time_ps, energy_values, label=model_name, color=color)
  plt.xlabel("Time (ps)")
  plt.ylabel("Potential Energy (eV)")
  plt.title("Potential Energy vs Step for CO$_2$ Simulations")
  plt.legend()
  plt.grid()
  plt.savefig(os.path.join(plot_folder, "potential_energy_vs_step.png"), dpi=400)
  plt.show()

  # Kinetic Energy vs Step Plot
  plt.figure(figsize=(8, 6))
  for model_name in steps:
    step_values = sorted(steps[model_name].keys())
    energy_values = [kinetic_energies[model_name][step] for step in step_values]
    color = model_colors.get(model_name, None)
    time_ps = np.array(step_values) * 0.01  # assuming 10 fs per step
    plt.plot(time_ps, energy_values, label=model_name, color=color)
  plt.xlabel("Time (ps)")
  plt.ylabel("Kinetic Energy (eV)")
  plt.title("Kinetic Energy vs Step for CO$_2$ Simulations")
  plt.legend()
  plt.grid()
  plt.show()

  # Velocity Distribution Plot
  plt.figure(figsize=(8, 6))
  for model_name in velocities:
    final_frame = max(velocities[model_name].keys())
    vel = velocities[model_name][final_frame]
    speeds = np.linalg.norm(vel, axis=1)
    color = model_colors.get(model_name, None)
    sns.kdeplot(speeds, label=model_name, fill=True, color=color)
  plt.xlabel("Speed")
  plt.ylabel("Density")
  plt.title("Velocity Distribution for CO$_2$ Simulations")
  plt.legend()
  plt.grid()
  plt.savefig(os.path.join(plot_folder, "velocity_distribution.png"), dpi=400)
  plt.show()

  # Force Distribution Plot
  plt.figure(figsize=(8, 6))
  for model_name in forces:
    final_frame = max(forces[model_name].keys())
    frc = forces[model_name][final_frame]
    magnitudes = np.linalg.norm(frc, axis=1)
    color = model_colors.get(model_name, None)
    sns.kdeplot(magnitudes, label=model_name, fill=True, color=color)
  plt.xlabel("Force Magnitude")
  plt.ylabel("Density")
  plt.title("Force Distribution for CO$_2$ Simulations")
  plt.legend()
  plt.grid()
  plt.show()


  def compute_msd(positions):
    n_atoms = positions.shape[0]
    n_frames = positions.shape[1]
    msd = np.zeros(n_frames)
    for dt in range(1, n_frames):
      diffs = positions[:, dt:, :] - positions[:, :-dt, :]
      sq_diffs = np.sum(diffs ** 2, axis=2)
      msd[dt] = np.mean(sq_diffs)
    return msd


  plt.figure(figsize=(8, 6))
  for model_name in positions:
    pos = np.array([positions[model_name][frame] for frame in sorted(positions[model_name].keys())])
    msd = compute_msd(pos)
    time = np.arange(len(msd))
    color = model_colors.get(model_name, None)
    plt.plot(time, msd, label=model_name, color=color)
  plt.xlabel("Time")
  plt.ylabel("Mean Squared Displacement (MSD)")
  plt.title("Mean Squared Displacement (MSD) for CO$_2$ Simulations")
  plt.legend()
  plt.grid()
  # plt.savefig(os.path.join(plot_folder, "msd_vs_time.png"))
  plt.show()

  from scipy.spatial.distance import pdist


  def compute_rdf(positions, box_length, bin_width=0.025, r_max=10.0):
    distances = pdist(positions)
    hist, bin_edges = np.histogram(distances, bins=np.arange(0, r_max + bin_width, bin_width))
    r = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    number_density = len(positions) / (box_length ** 3)
    norm = (4 * np.pi * r ** 2 * bin_width) * number_density * len(positions)
    rdf = hist / norm
    return rdf, r


  plt.figure(figsize=(8, 6))

  for model_name in positions:
    rdf_accum = None
    n_frames = 0
    # Average over frames after equilibration (e.g., skip first 1000 frames)
    frame_keys = sorted([f for f in positions[model_name] if f >= 1000])

    for f in tqdm(frame_keys):
      pos = positions[model_name][f]
      box_length = volumes[model_name][f] ** (1 / 3)
      rdf, r = compute_rdf(pos, box_length)
      if rdf_accum is None:
        rdf_accum = rdf
      else:
        rdf_accum += rdf
      n_frames += 1

    rdf_avg = rdf_accum / n_frames
    color = model_colors.get(model_name, None)
    plt.plot(r, rdf_avg, label=model_name, color=color)

  plt.xlabel("Distance r (Å)")
  plt.ylabel("g(r)")
  plt.title("Time-Averaged Radial Distribution Function (RDF) for CO₂ Simulations")
  plt.legend()
  plt.grid()
  plt.savefig(os.path.join(plot_folder, "rdf_co2.png"), dpi=400)
  plt.show()


  def compute_msd(positions_dict):
    model_msds = {}
    for model_name, frames in positions_dict.items():
      frame_indices = sorted(frames.keys())
      positions = np.array([frames[i] for i in frame_indices])
      n_frames, n_atoms, _ = positions.shape

      displacements = positions - positions[0]
      squared_displacements = np.sum(displacements ** 2, axis=2)  # per atom
      msd = np.mean(squared_displacements, axis=1)  # averaged over atoms

      model_msds[model_name] = (frame_indices, msd)
    return model_msds


  def fit_diffusion(msd_data, time_per_frame=1.0):
    diffusion_coeffs = {}
    fits = {}
    for model_name, (frames, msd) in msd_data.items():
      times = np.array(frames) * time_per_frame
      # Use only later frames (skip transient part)
      start_idx = len(times) // 3
      slope, intercept, r, p, stderr = linregress(times[start_idx:], msd[start_idx:])
      D = slope / 6.0
      diffusion_coeffs[model_name] = D
      fits[model_name] = (times, intercept + slope * times)
    return diffusion_coeffs, fits


  # Compute MSDs and fits
  msd_data = compute_msd(positions)
  diffusion_coeffs, fits = fit_diffusion(msd_data, time_per_frame=1.0)

  # Plot MSD for each model
  plt.figure(figsize=(8, 6))
  for model_name, (frames, msd) in msd_data.items():
    color = model_colors.get(model_name, None)
    msd_in_cubic_nanometers = msd * 0.01
    times_in_ps = np.array(frames) * 0.01  # assuming each frame is 10 fs
    plt.plot(times_in_ps, msd_in_cubic_nanometers, label=f"{model_name} MSD", color=color)
  plt.xlabel("Time (ps)")
  plt.ylabel("Mean Squared Displacement (nm²)")
  plt.title("MSD over Time")
  plt.legend()
  plt.grid()
  plt.tight_layout()
  plt.savefig(os.path.join(plot_folder, "msd_per_model.png"))
  plt.show()

  # Plot MSD + linear fits
  plt.figure(figsize=(8, 6))
  for model_name, (frames, msd) in msd_data.items():
    times = np.array(frames)
    color = model_colors.get(model_name, None)
    plt.plot(times, msd, label=f"{model_name} MSD", color=color)
    plt.plot(times, fits[model_name][1], '--', label=f"{model_name} linear fit", color=color)
  plt.xlabel("Time (arb. units)")
  plt.ylabel("MSD (Å²)")
  plt.title("MSD and Linear Fit for Diffusion Estimation")
  plt.legend()
  plt.grid()
  plt.tight_layout()
  plt.show()

  # Bar plot of diffusion coefficients
  plt.figure(figsize=(8, 5))
  models = list(diffusion_coeffs.keys())
  D_vals = [diffusion_coeffs[m] for m in models]
  colors = {m: model_colors.get(m, 'gray') for m in models}
  plt.bar(models, D_vals, color=[colors[m] for m in models])
  plt.ylabel("Diffusion Coefficient (Å²/frame)")
  plt.title("Estimated Diffusion Coefficients per Model")
  plt.xticks(rotation=45)
  plt.tight_layout()
  plt.grid()
  plt.show()

  # --- Define atomic masses (amu) for CO2 COM calculation ---
  MASS_C = 12.011
  MASS_O = 15.999
  # Create an array of masses in the order [C, O, O]
  ATOM_MASSES = np.array([MASS_C, MASS_O, MASS_O])
  M_TOTAL = np.sum(ATOM_MASSES)
  # Reshape weights for broadcasting later. Shape becomes (1, 3, 1)
  # This allows multiplying (n_frames, n_mols, 3, 3) * (1, 3, 1)
  ATOM_WEIGHTS = (ATOM_MASSES / M_TOTAL).reshape((1, 3, 1))


  def compute_msd_for_block(frames_dict, time_per_step):
    """
    Computes the MSD for a block of frames using the Center of Mass (COM)
    of each CO2 molecule.

    Assumes atom ordering is [C1, O1a, O1b, C2, O2a, O2b, ...].
    """
    # 1. Get positions and basic info
    frame_indices = sorted(frames_dict.keys())
    positions = np.array([frames_dict[i] for i in frame_indices])
    n_frames, n_atoms_total, dims = positions.shape

    # Need enough frames for both COM calc and good MSD stats
    if n_frames < 20:
      return np.array([]), np.array([])

    # 2. Calculate Center of Mass (COM) for each molecule
    n_molecules = n_atoms_total // 3
    if n_atoms_total % 3 != 0:
      print(f"Warning: Total atoms ({n_atoms_total}) is not divisible by 3.")
      # Attempt to truncate to the last complete molecule
      n_atoms_total = n_molecules * 3
      positions = positions[:, :n_atoms_total, :]

    # Reshape positions: (n_frames, n_atoms_total, 3) -> (n_frames, n_molecules, 3_atoms, 3_dims)
    try:
      mol_positions = positions.reshape((n_frames, n_molecules, 3, dims))
    except ValueError as e:
      print(f"Error reshaping positions: {e}")
      return np.array([]), np.array([])

    # Calculate COM using numpy broadcasting
    # (n_frames, n_mols, 3_atoms, 3_dims) * (1_frame, 3_atoms, 1_dim)
    com_positions = np.sum(mol_positions * ATOM_WEIGHTS, axis=2)
    # com_positions shape is now (n_frames, n_molecules, 3_dims)

    # 3. Determine time lags (same as before)
    if len(frame_indices) < 2:
      return np.array([]), np.array([])  # Need at least 2 frames

    steps_per_frame = frame_indices[1] - frame_indices[0]
    time_per_frame_fs = steps_per_frame * time_per_step

    # 4. Compute MSD on COM positions
    max_lag_index = n_frames // 4  # Use 1/4 of data for good stats
    if max_lag_index < 1:
      return np.array([]), np.array([])

    msd_values = []
    time_lags = []

    for lag_idx in range(1, max_lag_index + 1):
      # Displacements of COMs
      displacements = com_positions[lag_idx:] - com_positions[:-lag_idx]

      # Squared displacements of COMs
      sq_displacements = np.sum(displacements ** 2, axis=2)  # Shape: (n_frames-lag, n_molecules)

      # Average over all molecules AND all time origins (t0)
      msd_for_lag = np.mean(sq_displacements)

      msd_values.append(msd_for_lag)
      time_lags.append(lag_idx * time_per_frame_fs)

    return np.array(time_lags), np.array(msd_values)


  def fit_diffusion(times, msd):
    """
    Fits the diffusion coefficient from MSD vs. time data.
    (Unchanged)
    """
    if len(times) < 5:  # Not enough points to fit
      return np.nan

    start_idx = len(times) // 10
    end_idx = len(times) // 2

    if end_idx - start_idx < 3:
      start_idx = 0
      end_idx = len(times)
      if end_idx < 2:
        return np.nan

    slope, intercept, r, p, stderr = linregress(times[start_idx:end_idx], msd[start_idx:end_idx])

    # D = slope / 6 (for 3D)
    D = slope / 6.0
    return D


  def compute_temperature_dependent_diffusion(
    positions, temperatures,
    steps_per_block=1000,
    time_per_step=10.0
  ):
    """
    Computes diffusion coefficients vs. temperature from trajectory blocks.
    (Unchanged)
    """
    D_vs_T = defaultdict(list)
    for model_name, frames in positions.items():
      if not frames:
        continue
      max_frame = max(frames.keys())
      n_blocks = max_frame // steps_per_block

      for i in tqdm(range(n_blocks)):
        block_start = i * steps_per_block
        block_end = (i + 1) * steps_per_block

        block_frames = {k: v for k, v in frames.items() if block_start <= k < block_end}

        if len(block_frames) < 50:
          continue

        # This now calls the new COM-based MSD function
        times, msd = compute_msd_for_block(block_frames, time_per_step)

        if len(times) == 0:
          continue

        D = fit_diffusion(times, msd) * 10# Convert from Å²/fs to cm²/s
        if np.isnan(D):
          continue

        T_vals = [temperatures[model_name][f] for f in block_frames if f in temperatures[model_name]]
        if not T_vals:
          continue

        T_mean = np.mean(T_vals)
        D_vs_T[model_name].append((T_mean, D))

    return D_vs_T


  # Compute diffusion per temperature block
  D_vs_T = compute_temperature_dependent_diffusion(positions, temperatures, steps_per_block=1000)

  # Plot D(T)
  plt.figure(figsize=(8, 6))
  for model_name, vals in D_vs_T.items():
    vals = sorted(vals, key=lambda x: x[0])
    temps = [v[0] for v in vals]
    Ds = [v[1] for v in vals]
    color = model_colors.get(model_name, None)
    plt.plot(temps, Ds, 'o-', label=model_name, color=color)
    # Draw exponential fit lines
    if len(temps) >= 2:
      # log_Ds = np.log(Ds)
      # slope, intercept, r, p, stderr = linregress(temps, log_Ds)
      # fit_Ds = np.exp(intercept + slope * np.array(temps))
      # plt.plot(temps, fit_Ds, '--', color=color)

      # ln(D) = A + B/T + C·ln(T).
      ln_Ds = np.log(Ds)
      inv_Ts = 1.0 / np.array(temps)
      ln_Ts = np.log(temps)

      # Build design matrix with a constant term (for A)
      X = np.column_stack((np.ones_like(inv_Ts), inv_Ts))#, ln_Ts))

      # Solve for A, B, C using least squares
      A, B = np.linalg.lstsq(X, ln_Ds, rcond=None)[0]

      fit_ln_Ds = A + B * inv_Ts# + C * ln_Ts
      fit_Ds = np.exp(fit_ln_Ds)
      plt.plot(temps, fit_Ds, '--', color=color)

      # Plot A, B, C parameters in the legend
      # plt.text(temps[-1], fit_Ds[-1], f"A={A:.2f}\nB={B:.2f}", color=color, fontsize=8,
      #          verticalalignment='bottom')
      # Write A, B values to console
      print(f"{model_name} fit parameters: A={A:.4f}, B={B:.4f}")

  # Print diffusion coefficients for T=300K
    if 300 in temps:
      D_300K = Ds[temps.index(300)]
      print(f"{model_name} Diffusion Coefficient at 300K: {D_300K:.4e} cm²/s")

  D_ref = 0.120
  A_ref = 0.015
  B_ref = -637
  T_fit = np.linspace(233, 363, 100)
  ln_D_ref = A_ref + B_ref / T_fit
  D_fit = np.exp(ln_D_ref)
  plt.plot(T_fit, D_fit, 'k--', label="Reference", alpha=1)

  plt.xlabel("Temperature (K)")
  plt.ylabel("Diffusion Coefficient (cm²/s)")
  plt.title("Diffusion Coefficient vs Temperature")
  plt.legend()
  plt.grid()
  plt.tight_layout()
  plt.savefig(os.path.join(plot_folder, "diffusion_vs_temperature.png"), dpi=400)
  plt.show()

  print("Done!")
