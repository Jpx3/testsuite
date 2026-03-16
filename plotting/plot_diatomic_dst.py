import pickle
import matplotlib.pyplot as plt
import numpy as np
import os

folder = "../results/diatomic_energy_curve/"

files = [f for f in os.listdir(folder) if f.endswith(".pkl")]

# Find common elements across files
elements = None
for file in files:
  dat = pickle.load(open(os.path.join(folder, file), "rb"))
  if elements is None:
    elements = set(dat.keys())
  else:
    elements &= set(dat.keys())

print(f"Common elements across all files: {elements}")

model_diffs = {}
distances = None

for element in elements:
  dft_curve = None
  element_curves = {}

  # Load mean curves
  for file in files:
    name = file.replace(".pkl", "")
    dat = pickle.load(open(os.path.join(folder, file), "rb"))
    rotation_curves = dat[element]

    # if name.lower().startswith("orb_v3_direct"):
    #   continue

    n_rots = len(rotation_curves)
    n_dist = len(rotation_curves[0]["distances"])
    data = np.zeros((n_rots, n_dist))

    for i, curve in enumerate(rotation_curves):
      data[i, :] = curve["energies"]

    mean_curve = np.mean(data, axis=0)

    if name.lower().startswith("dft"):
      dft_curve = mean_curve
      distances = np.array(rotation_curves[0]["distances"])
    else:
      element_curves[name] = mean_curve

  # Compute diffs for this element
  if dft_curve is not None:
    valid_mask = dft_curve != 0  # ignore points where DFT == 0
    for name, curve in element_curves.items():
      diff = np.full_like(dft_curve, np.nan)  # start with NaN
      diff[valid_mask] = curve[valid_mask] - dft_curve[valid_mask]
      if name not in model_diffs:
        model_diffs[name] = []
      model_diffs[name].append(diff)

# Average across elements (ignoring NaNs where DFT was 0)
plt.figure(figsize=(8, 6))
from model_colors import model_colors

for name, diffs in model_diffs.items():
  diffs = np.array(diffs)  # shape: (n_elements, n_distances)
  mean_diff = np.nanmean(diffs, axis=0)
  std_diff = np.nanstd(diffs, axis=0)
  line = plt.plot(distances, mean_diff, label=name, color=model_colors.get(name, None))[0]
  plt.fill_between(
    distances, mean_diff - std_diff, mean_diff + std_diff, alpha=0.2, color=line.get_color()
  )

plt.axhline(0, color='black', linestyle='--', linewidth=1)
plt.xlabel("Interatomic Distance (Å)")
plt.ylabel("Mean Energy Difference to DFT (eV)")
plt.title("Average Model vs DFT Energy Differences (excluding noble gases)")
plt.tight_layout()
plt.legend()
plt.ylim(-150, 75)
os.makedirs("figures/diatomic_overall", exist_ok=True)
plt.savefig("figures/diatomic_overall/all-merged.png", dpi=300)
plt.show()

plt.close()

plt.figure(figsize=(8, 6))
# Per-model plots
for name, diffs in model_diffs.items():
  diffs = np.array(diffs)  # shape: (n_elements, n_distances)
  mean_diff = np.nanmean(diffs, axis=0)
  std_diff = np.nanstd(diffs, axis=0)

  plt.figure(figsize=(8, 6))
  line = plt.plot(distances, mean_diff, label=name, color=model_colors.get(name, None))[0]
  plt.fill_between(distances, mean_diff - std_diff, mean_diff + std_diff, alpha=0.2, color=line.get_color())

  plt.axhline(0, color='black', linestyle='--', linewidth=1)
  plt.xlabel("Interatomic Distance (Å)")
  plt.ylabel("Mean Energy Difference to DFT (eV)")
  plt.title(f"Model vs DFT Energy Differences: {name} (excluding noble gases)")
  plt.tight_layout()
  plt.legend()
  plt.ylim(-150, 75)
  os.makedirs("figures/diatomic_overall", exist_ok=True)
  plt.savefig(f"figures/diatomic_overall/{name}-merged.png", dpi=300)
  plt.show()
  plt.close()
