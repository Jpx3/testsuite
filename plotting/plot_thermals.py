import pickle
import re
from typing import Optional

import phonopy
from phonopy import Phonopy
from tqdm import tqdm

global_thermals = dict()
global_frequencies = dict()
global_bands = dict()
global_dos = dict()
phonons = dict()

# list files in /mdr-phonondb/data
import os

data_dir = "../tests/mdr-phonondb/data"
cache_dir = "../tests/mdr-phonondb/cache"
files = os.listdir(data_dir)

# Generate temp directory
import tempfile
import shutil

temp_dir = tempfile.mkdtemp()
print(f"Using temporary directory: {temp_dir}")

results = dict()

phonons_cache = os.path.join(cache_dir, "phonon_objects.pkl")
if os.path.exists(phonons_cache):
  phonons = pickle.load(open(phonons_cache, "rb"))
  print(f"Loaded cached results from {phonons_cache}")
else:
  print(f"No cache found at {phonons_cache}, preparing phonon objects from {data_dir}")
  import random

  # Fixed seed for deterministic behavior
  sparse = random.Random(1337)

  for file in tqdm(files):
    if file.endswith(".zip"):
      if sparse.random() < 0.95:
        continue  # Process only 5% of files for speed

      import zipfile

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

folder = "../results/phonon_norelax/"
files = os.listdir(folder)

for file in tqdm(files):
  if file.endswith(".pkl"):
    name = file.replace(".pkl", "")
    dat = pickle.load(open(os.path.join(folder, file), "rb"))

    n_files = 0
    for key in dat.keys():
      res = dat[key]["results"]
      frequencies = res[0]
      thermal_props = res[1]
      band_structure = res[2]
      total_dos = res[3]

      if global_frequencies.get(name) is None:
        global_frequencies[name] = dict()
      global_frequencies[name][key] = frequencies

      temps, fe, entropy, cv = thermal_props._thermal_properties
      # global_thermals[name] = (temps, fe, entropy, cv)
      if global_thermals.get(name) is None:
        global_thermals[name] = dict()
      global_thermals[name][key] = (temps, fe, entropy, cv)

      if global_bands.get(name) is None:
        global_bands[name] = dict()
      global_bands[name][key] = band_structure

      if global_dos.get(name) is None:
        global_dos[name] = dict()
      global_dos[name][key] = total_dos

# Verify all models have the same keys
all_keys = set()
for model in global_frequencies.keys():
  all_keys.update(global_frequencies[model].keys())
for model in global_frequencies.keys():
  model_keys = set(global_frequencies[model].keys())
  if model_keys != all_keys:
    # Print missing keys
    missing = all_keys - model_keys
    for m in missing:
      print(f"Model {model} is missing key {m}")

# Save all keys to a txt file if it doesn't exist
os.makedirs("../tests/mdr-phonondb", exist_ok=True)
if not os.path.exists("../tests/mdr-phonondb/used.txt"):
  with open("../tests/mdr-phonondb/used.txt", "w") as f:
    for key in all_keys:
      f.write(f"{key}\n")

reference = ["dft_gpaw_pw"]

# horizontal violin plot of "max frequency" deviation from reference structure
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import ptitprince as pt

os.makedirs("../plotting/figures/thermal", exist_ok=True)


def plot_horizontal_violin(
  df: pd.DataFrame,
  num_col: str,
  cat_col: str,
  title: str = "Violin Plot",
  z_score_threshold: float = 3.0,
  figsize: tuple = (8, 6),
  palette="viridis",
  save_path: str = None
):
  data_series = df[num_col]
  mean_val = data_series.mean()
  std_val = data_series.std()
  lower_limit = mean_val - z_score_threshold * std_val
  upper_limit = mean_val + z_score_threshold * std_val
  plt.figure(figsize=figsize)
  sns.violinplot(
    x=num_col, y=cat_col, data=df, scale="width",
    palette=palette, legend=False, gridsize=250, saturation=1
  )
  plt.title(title, fontsize=16)
  plt.xlabel(num_col, fontsize=12)
  plt.ylabel(cat_col, fontsize=12)
  plt.grid(axis='x', linestyle='--', alpha=0.7)
  plt.xlim(lower_limit, upper_limit)
  plt.axvline(0, color='k', linestyle='--')
  plt.tight_layout()
  if save_path:
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {save_path}")
  plt.show()


def plot_raincloud(
  df: pd.DataFrame,
  num_col: str,
  cat_col: str,
  title: str = "You forgot me :(",
  z_score_threshold: float = 3.0,
  figsize: tuple = (12, 8),
  save_path: str = None,
):
  def alphanumeric_key(s):
    def try_int(text):
      return int(text) if text.isdigit() else text.lower()

    return [try_int(c) for c in re.split('([0-9]+)', str(s))]

  unique_categories = df[cat_col].unique()
  sorted_categories = sorted(unique_categories, key=alphanumeric_key)

  df_plot = df.copy()
  df_plot[cat_col] = pd.Categorical(
    df_plot[cat_col], categories=sorted_categories, ordered=True
  )
  data_series = df[num_col]

  from model_colors import model_colors
  mypal = [model_colors[m] for m in df_plot[cat_col].cat.categories if m in model_colors]

  mean_val = data_series.mean()
  std_val = data_series.std()
  lower_limit = mean_val - z_score_threshold * std_val
  upper_limit = mean_val + z_score_threshold * std_val
  fig, ax = plt.subplots(figsize=figsize)
  pt.RainCloud(
    x=cat_col,
    y=num_col,
    data=df_plot,
    palette=mypal,
    orient='h',  # Use 'h' for horizontal orientation
    ax=ax
  )
  plt.title(title, fontsize=16, pad=20)
  # plt.ylabel(cat_col, fontsize=12)
  plt.yticks([])
  plt.ylabel('')
  plt.xlabel(num_col, fontsize=12)
  smaller_limit = min(abs(lower_limit), abs(upper_limit))
  lower_limit = -smaller_limit
  upper_limit = smaller_limit
  plt.xlim(lower_limit, upper_limit)
  plt.axvline(0, color='k', linestyle='--')
  plt.grid(axis='x', linestyle='--', alpha=0.7)
  plt.tight_layout()

  if save_path:
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {save_path}")

  plt.show()

from matplotlib.patches import Patch
def plot_legend(cat_col_order):
  from model_colors import model_colors
  handles = [
    Patch(color=model_colors[m], label=m)
    for m in cat_col_order
    if m in model_colors
  ]

  fig, ax = plt.subplots(figsize=(8, 1))  # short, wide figure
  ax.legend(
    handles=handles,
    ncol=3,
    frameon=False,
    loc='center',
    fontsize=10
  )
  ax.axis('off')  # hide axes
  plt.tight_layout()
  plt.savefig("../plotting/figures/thermal/legend.png", dpi=300, bbox_inches='tight')
  plt.show()


def plot_max_freq_deviation(global_frequencies, reference):
  data = []
  for name in global_frequencies.keys():
    if name in reference:
      continue
    for key in global_frequencies[name].keys():
      freqs = global_frequencies[name][key]
      max_freq = np.max(freqs)
      ref_max_freq = np.max(global_frequencies[reference[0]][key])
      max_dev = max_freq - ref_max_freq
      data.append((name, max_dev))

  df = pd.DataFrame(data, columns=["Model", "Max Frequency Deviation (cm$^{-1}$)"])

  plot_raincloud(
    df,
    num_col="Max Frequency Deviation (cm$^{-1}$)",
    cat_col="Model",
    title="Max Frequency Deviation",
    z_score_threshold=1.0,
    figsize=(6, 8),
    save_path="../plotting/figures/thermal/max_freq_deviation_raincloud.png"
  )


def plot_thermals(global_thermals, reference):
  fe_data = []
  ent_data = []
  cv_data = []
  for model_name in global_thermals.keys():
    if model_name in reference:
      continue
    for structure_name in global_thermals[model_name].keys():
      phon_obj: Phonopy = phonons.get(structure_name, None)
      if phon_obj is None:
        raise ValueError(f"Phonopy object for {structure_name} not found in phonons dictionary.")
      n_atoms = len(phon_obj.supercell.positions)
      temps, fe, entropy, cv = global_thermals[model_name][structure_name]
      # for idx, temperature in enumerate(temps):
      temperature = 300  # Focus on 300K
      idx = np.argmin(np.abs(temps - temperature))

      fe_dev = fe[idx] - global_thermals[reference[0]][structure_name][1][idx]
      # Fe is in kJ/mol
      fe_dev = fe_dev
      # Cap between -1000 and 1000
      if fe_dev < -400:
        fe_dev = -400
      if fe_dev > 200:
        fe_dev = 200

      ent_dev = entropy[idx] - global_thermals[reference[0]][structure_name][2][idx]
      # Entropy is kB per unit cell, convert to J/mol/K
      # ent_dev = ent_dev * 8.31446261815324 / n_atoms
      # Cap between -1000 and 1000
      if ent_dev < -500:
        ent_dev = -500
      if ent_dev > 1000:
        ent_dev = 1000

      cv_dev = cv[idx] - global_thermals[reference[0]][structure_name][3][idx]

      # Cv is kB per unit cell, convert to meV/atom/K

      fe_data.append((model_name, fe_dev))
      ent_data.append((model_name, ent_dev))
      cv_data.append((model_name, cv_dev))

  fe_df = pd.DataFrame(fe_data, columns=["Model", f"Free Energy Deviation (kJ/mol)"])
  ent_df = pd.DataFrame(ent_data, columns=["Model", f"Entropy Deviation (J/mol/K)"])
  cv_df = pd.DataFrame(cv_data, columns=["Model", f"Heat Capacity Deviation (J/mol/K)"])

  figsize = (6, 8)

  plot_raincloud(
    fe_df,
    num_col=f"Free Energy Deviation (kJ/mol)",
    cat_col="Model",
    title="Free Energy Deviation",
    z_score_threshold=1.0,
    figsize=figsize,
    save_path="../plotting/figures/thermal/fe_deviation_raincloud.png"
  )

  plot_raincloud(
    ent_df,
    num_col=f"Entropy Deviation (J/mol/K)",
    cat_col="Model",
    title="Entropy Deviation",
    z_score_threshold=1.0,
    figsize=figsize,
    save_path="../plotting/figures/thermal/entropy_deviation_raincloud.png"
  )

  plot_raincloud(
    cv_df,
    num_col=f"Heat Capacity Deviation (J/mol/K)",
    cat_col="Model",
    title="Heat Capacity Deviation",
    z_score_threshold=1.0,
    figsize=figsize,
    save_path="../plotting/figures/thermal/cv_deviation_raincloud.png"
  )


plot_max_freq_deviation(global_frequencies, reference)
plot_thermals(global_thermals, reference)
plot_legend(sorted(global_frequencies.keys()))