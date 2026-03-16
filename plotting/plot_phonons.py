import os
import pickle
import random
import shutil
import tempfile
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import phonopy
from phonopy import Phonopy
from phonopy.phonon.band_structure import BandStructure
from tqdm import tqdm

# --- Data Loading and Caching (Original code is correct) ---

data_dir = "../tests/mdr-phonondb/data"
cache_dir = "../tests/mdr-phonondb/cache"
files = os.listdir(data_dir)

# Generate temp directory
temp_dir = tempfile.mkdtemp()
print(f"Using temporary directory: {temp_dir}")

phonons = dict()
global_bands = dict()
global_dos = dict()

phonons_cache = os.path.join(cache_dir, "phonon_objects.pkl")
if os.path.exists(phonons_cache):
  phonons = pickle.load(open(phonons_cache, "rb"))
  print(f"Loaded cached results from {phonons_cache}")
else:
  print(f"No cache found at {phonons_cache}, preparing phonon objects from {data_dir}")

  # Fixed seed for deterministic behavior
  sparse = random.Random(1337)

  for file in tqdm(files):
    if file.endswith(".zip"):
      if sparse.random() < 0.95:
        continue  # Process only 5% of files for speed

      with zipfile.ZipFile(os.path.join(data_dir, file), 'r') as zip_ref:
        name = file[:-4]
        file_bytes = zip_ref.read("phonopy_params.yaml.xz")
        # Write bytes to a temporary file
        temp_file_path = os.path.join(temp_dir, f"{name}_phonopy_params.yaml.xz")
        try:
          with open(temp_file_path, 'wb') as temp_file:
            temp_file.write(file_bytes)
          p = phonopy.load(temp_file_path, produce_fc=False)
          if len(p.supercells_with_displacements) < 50:
            phonons[name] = p
        except Exception as e:
          print(f"Error processing {name}: {e}")
        finally:
          # Ensure the temporary file is deleted after use
          if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

  shutil.rmtree(temp_dir)
  print(f"Removed temporary directory: {temp_dir}")
  print(f"Processed {len(phonons)} phonon objects from {data_dir}")
  os.makedirs(cache_dir, exist_ok=True)
  pickle.dump(phonons, open(phonons_cache, "wb"))
  print(f"Saved cached results to {phonons_cache}")

# --- Result Processing (Original code is correct) ---

folder = "../results/phonon_norelax/"
files = os.listdir(folder)
first_key = None

# Plot phonon band structures per model
for file in tqdm(files):
  if file.endswith(".pkl"):
    name = file.replace(".pkl", "")
    dat = pickle.load(open(os.path.join(folder, file), "rb"))

    for key in dat.keys():
      if first_key is None:
        first_key = key
      res = dat[key]["results"]
      freqs = res[0]
      band_structure = res[2]
      total_dos = res[3]

      if global_bands.get(name) is None:
        global_bands[name] = dict()
      global_bands[name][key] = band_structure

      if global_dos.get(name) is None:
        global_dos[name] = dict()
      global_dos[name][key] = total_dos


class BandPlot:
  def __init__(self, axs):
    self._axs = axs
    self.xscale = None
    self._decorated = False

  def plot(
    self,
    distances,
    frequencies,
    path_connections,
    fmt=None,
    label=None
  ):
    if fmt is None:
      _fmt = "r-"
    else:
      _fmt = fmt

    if self.xscale is None:
      self.set_xscale_from_data(frequencies, distances)

    count = 0
    distances_scaled = [d * self.xscale for d in distances]
    for i, (d, f, c) in enumerate(
      zip(distances_scaled, frequencies, path_connections)
    ):
      ax = self._axs[count]
      if i == 0 and label is not None:
        curves = ax.plot(d, f, _fmt, linewidth=1)
        curves[0].set_label(label)
        ax.legend()
      else:
        ax.plot(d, f, _fmt, linewidth=1)
      if not c:
        count += 1

  def plot_dual(
    self,
    orig_frequencies,
    ref_distances,
    ref_frequencies,
    ref_path_connections,
    model_color="red",
    fmt=None,
    label=None
  ):
    # if fmt is None:
    #   _fmt = "r-"
    # else:
    #   _fmt = fmt
    # _fmt = None  # use default color cycle

    if self.xscale is None:
      self.set_xscale_from_data(ref_frequencies, ref_distances)

    count = 0
    distances_scaled = [d * self.xscale for d in ref_distances]
    for i, (d, f, c) in enumerate(
      zip(distances_scaled, ref_frequencies, ref_path_connections)
    ):
      ax = self._axs[count]
      if i == 0 and label is not None:
        curves = ax.plot(d, f, linewidth=1, color="black")
        curves[0].set_label(label)
        ax.legend()
      else:
        ax.plot(d, f, linewidth=1, color="black")
      if not c:
        count += 1

    count = 0
    for i, (d, f, c) in enumerate(
      zip(distances_scaled, orig_frequencies, ref_path_connections)
    ):
      ax = self._axs[count]
      if i == 0 and label is not None:
        curves = ax.plot(d, f, linewidth=1, color=model_color)
        curves[0].set_label(label)
        ax.legend()
      else:
        ax.plot(d, f, linewidth=1, color=model_color)
      if not c:
        count += 1

  def set_xscale_from_data(self, frequencies, distances):
    max_freq = max([np.max(fq) for fq in frequencies])
    max_dist = distances[-1][-1]
    self.xscale = max_freq / max_dist * 1.5

  def decorate(self, labels, path_connections, frequencies, distances):
    if self._decorated:
      raise RuntimeError("Already BandPlot instance is decorated.")
    else:
      self._decorated = True

    if self.xscale is None:
      self.set_xscale_from_data(frequencies, distances)

    distances_scaled = [d * self.xscale for d in distances]

    # T T T F F -> [[0, 3], [4, 4]]
    lefts = [0]
    rights = []
    for i, c in enumerate(path_connections):
      if not c:
        lefts.append(i + 1)
        rights.append(i)
    seg_indices = [list(range(lft, rgt + 1)) for lft, rgt in zip(lefts, rights)]
    special_points = []
    for indices in seg_indices:
      pts = [distances_scaled[i][0] for i in indices]
      pts.append(distances_scaled[indices[-1]][-1])
      special_points.append(pts)

    self._axs[0].set_ylabel("Frequency (THz)")
    l_count = 0
    for ax, spts in zip(self._axs, special_points):
      ax.xaxis.set_ticks_position("both")
      ax.yaxis.set_ticks_position("both")
      ax.xaxis.set_tick_params(which="both", direction="in")
      ax.yaxis.set_tick_params(which="both", direction="in")
      ax.set_xlim(spts[0], spts[-1])
      ax.set_xticks(spts)
      if labels is None:
        ax.set_xticklabels(
          [
            "",
          ]
          * len(spts)
        )
      else:
        ax.set_xticklabels(labels[l_count: (l_count + len(spts))])
        l_count += len(spts)
      ax.plot(
        [spts[0], spts[-1]], [0, 0], linestyle=":", linewidth=0.5, color="b"
      )


def plot_band_structure_manual(
  band_structure: BandStructure,
  reference_band_structure: BandStructure = None,
  color="red",
  model_name=None
):
  if reference_band_structure is None:
    raise ValueError("Reference band structure must be provided for manual plotting.")

  from mpl_toolkits.axes_grid1 import ImageGrid

  n = len([x for x in band_structure.path_connections if not x])
  fig = plt.figure()
  axs = ImageGrid(
    fig,
    111,  # similar to subplot(111)
    nrows_ncols=(1, n),
    axes_pad=0.11,
    label_mode="L",
  )

  # ref labels must match band_structure labels
  if not np.array_equal(reference_band_structure.distances, band_structure.distances):
    raise ValueError("Reference and target band structures must have the same labels.")

  # ref distances must match band_structure distances
  if not np.array_equal(reference_band_structure.distances, band_structure.distances):
    raise ValueError("Reference and target band structures must have the same distances.")

  # ref path_connections must match band_structure path_connections
  if not np.array_equal(reference_band_structure.path_connections, band_structure.path_connections):
    raise ValueError("Reference and target band structures must have the same path connections.")

  # band_structure.plot(ax=axs)
  bp = BandPlot(axs)
  bp.decorate(
    reference_band_structure.labels,
    reference_band_structure.path_connections,
    reference_band_structure.frequencies,
    reference_band_structure.distances
  )
  bp.plot_dual(
    band_structure.frequencies,
    reference_band_structure.distances,
    reference_band_structure.frequencies,
    reference_band_structure.path_connections,
    model_color=color,
  )

  if model_name is not None:
    fig.suptitle(f"Example Phonon Band Structure - {model_name}")


# (The plot_dos function can remain as is)

# Create a directory to save the output plots
output_dir = "figures/phonon_plots"
os.makedirs(output_dir, exist_ok=True)
print(f"Saving plots to '{output_dir}/'")

collect_first_bands = dict()

reference_first = None

for name in global_bands.keys():
  bands = global_bands[name]
  dos = global_dos[name]

  if first_key not in bands:
    print(f"Warning: Key '{first_key}' not found in model '{name}'. Skipping.")
    continue

  if name != "dft_gpaw_pw":
    continue

  print(f"Using '{name}' as reference for key '{first_key}'")

  if reference_first is None:
    reference_first = bands[first_key]

  collect_first_bands[name] = bands[first_key]

for name in global_bands.keys():
  bands = global_bands[name]
  dos = global_dos[name]

  # phononnnn: Phonopy = phonons[name]
  # phononnnn.get_band_structure_dict()
  if first_key not in bands:
    print(f"Warning: Key '{first_key}' not found in model '{name}'. Skipping.")
    continue

  collect_first_bands[name] = bands[first_key]

  if name == "dft_gpaw_pw":
    continue  # Skip reference plot here

  import model_colors

  color = model_colors.model_colors[name]

  myphonon = phonons[first_key]

  retdict = {
    "distances": bands[first_key].distances,
    "qpoints": bands[first_key].qpoints,
    "eigenvalues": bands[first_key].frequencies,
    "vectors": bands[first_key].eigenvectors,
    "group_velocities": bands[first_key].group_velocities,
    "natoms": len(myphonon.unitcell),
    "atom_numbers": myphonon.unitcell.numbers,
    "atom_types": myphonon.unitcell.symbols,
    "lattice": myphonon.unitcell.cell,
    "atom_pos_car": myphonon.unitcell.positions,
    "atom_pos_red": myphonon.unitcell.scaled_positions,
    "repetitions": [1, 1, 1],
    "name": "Graphene",
    "highsym_qpts": [[0, ""], [20, ""], [30, ""], [50, ""]]
  }

  def convert_numpy_to_list(obj):
    if isinstance(obj, np.ndarray):
      return obj.tolist()
    elif isinstance(obj, dict):
      return {k: convert_numpy_to_list(v) for k, v in obj.items()}
    elif isinstance(obj, list):
      return [convert_numpy_to_list(v) for v in obj]
    else:
      return obj

  retdict = convert_numpy_to_list(retdict)
  import json

  json_path = os.path.join(output_dir, f"band_structure_{name}.json")
  with open(json_path, "w") as f:
    json.dump(retdict, f, indent=2)

  fig, ax = plt.subplots(figsize=(8, 6))
  plot_band_structure_manual(
    bands[first_key], color=color,
    reference_band_structure=reference_first,
    model_name=name
  )
  ax.set_title(f"Phonon Band Structure - {name} ({first_key})")
  plt.tight_layout()
  plt.savefig(os.path.join(output_dir, f"band_structure_{name}.png"), dpi=300)
  plt.close(fig)

# plt.savefig(os.path.join(output_dir, f"band_structure_overlay_{first_key}.png"))
plt.show()
# plt.close(fig)
