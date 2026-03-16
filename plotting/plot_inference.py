import pickle
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
  # folder = "../results/inference_time_node21_2/"
  folder = "../results/inference_time/"
  import os
  files = os.listdir(folder)
  model_names = [f.replace(".pkl", "") for f in files if f.endswith(".pkl")]
  print("Models found:", model_names)

  from model_colors import model_colors
  mypal = [model_colors.get(m, "#000") for m in model_names]

  for type in ["gpu", "cpu"]:
    plt.figure(figsize=(8, 6))
    for file in files:
      if file.endswith(".pkl"):
        name = file.replace(".pkl", "")
        dat = pickle.load(open(os.path.join(folder, file), "rb"))

        if type not in dat["results"]:
          # print(f"Skipping {name} for {type} inference (data not found).")
          continue

        dat = dat["results"][type]
        n_atoms = dat["n_atoms"]
        times = dat["times_ms"]

        # Group times by n_atoms
        groups = defaultdict(list)
        for n, t in zip(n_atoms, times):
          groups[n].append(t/1000)

        n_atoms_sorted = sorted(groups.keys())
        relative_medians = []
        relative_err_low = []
        relative_err_high = []

        for n in n_atoms_sorted:
          t_list = groups[n]
          median = np.median(t_list) # Convert to seconds
          p25, p75 = np.percentile(t_list, [25, 75])

          # Relative to number of atoms
          relative_medians.append(median)
          relative_err_low.append(median - p25)
          relative_err_high.append(p75 - median)

        # Asymmetric error bars
        yerr = [relative_err_low, relative_err_high]

        plt.errorbar(
          n_atoms_sorted,
          relative_medians,
          yerr=yerr,
          label=name,
          capsize=5,
          color=model_colors.get(name, None)
        )

    plt.xscale("log")
    plt.yscale("log")
    # 0.1, 1, 10
    plt.xticks([10, 100, 1000, 10000, 100000], ["10", "100", "1k", "10k", "100k"])
    plt.yticks([0.1, 1, 10, 100], ["0.1", "1", "10", "100"])
    plt.xlabel("Number of atoms")
    plt.ylabel("Inference time (s)")
    plt.title("Inference Time vs Number of Atoms for " + type.upper() + " inference")
    plt.legend()
    plt.grid()
    # Save to 'plotting/figures/inference_time/absolute_inference_time.png'
    if not os.path.exists("figures/inference_time/"):
      os.makedirs("figures/inference_time/")
    plt.savefig(f"figures/inference_time/absolute_inference_time_{type}.png", dpi=300)
    plt.show()

    plt.figure(figsize=(8, 6))
    # Relative inference time
    for file in files:
      if file.endswith(".pkl"):
        name = file.replace(".pkl", "")
        dat = pickle.load(open(os.path.join(folder, file), "rb"))

        if type not in dat["results"]:
          # print(f"Skipping {name} for {type} inference (data not found).")
          continue

        dat = dat["results"][type]
        n_atoms = dat["n_atoms"]
        times = dat["times_ms"]

        # Group times by n_atoms
        groups = defaultdict(list)
        for n, t in zip(n_atoms, times):
          groups[n].append(t)

        n_atoms_sorted = sorted(groups.keys())
        relative_medians = []
        relative_err_low = []
        relative_err_high = []

        for n in n_atoms_sorted:
          t_list = groups[n]
          median = np.median(t_list)
          p25, p75 = np.percentile(t_list, [25, 75])

          # Relative to number of atoms
          relative_medians.append(median / n)
          relative_err_low.append((median - p25) / n)
          relative_err_high.append((p75 - median) / n)

        # Asymmetric error bars
        yerr = [relative_err_low, relative_err_high]

        plt.errorbar(
          n_atoms_sorted,
          relative_medians,
          yerr=yerr,
          label=name,
          capsize=5,
          color=model_colors.get(name, None)
        )

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Number of atoms")
    plt.ylabel("Relative inference time (ms/atom)")
    plt.xticks([10, 100, 1000, 10000, 100000], ["10", "100", "1k", "10k", "100k"])
    plt.yticks([0.001, 0.01, 0.1, 1, 10], ["0.001", "0.01", "0.1", "1", "10"])
    plt.title("Relative Inference Time vs Number of Atoms for " + type.upper() + " inference")
    plt.legend()
    plt.grid()
    plt.savefig(f"figures/inference_time/relative_inference_time_{type}.png", dpi=300)
    plt.show()

  # ns/day table
    print(f"\n=== {type.upper()} Inference Speed (ns/day) ===")
    print(f"{'Model':<20} {'10 atoms':<15} {'100 atoms':<15} {'1k atoms':<15} {'10k atoms':<15} {'100k atoms':<15} ")
    for file in files:
      if file.endswith(".pkl"):
        name = file.replace(".pkl", "")
        dat = pickle.load(open(os.path.join(folder, file), "rb"))

        if type not in dat["results"]:
          # print(f"Skipping {name} for {type} inference (data not found).")
          continue

        dat = dat["results"][type]
        n_atoms = dat["n_atoms"]
        times = dat["times_ms"]

        # Group times by n_atoms
        groups = defaultdict(list)
        for n, t in zip(n_atoms, times):
          groups[n].append(t)  # in ms

        # ns/day calculation
        ns_per_day = {}
        for n in [10, 100, 1000, 10000, 100000]:
          if n in groups:
            mean_time_ms = np.mean(groups[n])
            ns_per_day[n] = 86.4 / mean_time_ms
          else:
            ns_per_day[n] = None
        print(f"{name:<20} "
              f"{ns_per_day[10]:<15.2f} "
              f"{ns_per_day[100]:<15.2f} "
              f"{ns_per_day[1000]:<15.2f} "
              f"{ns_per_day[10000]:<15.2f} "
              f"{ns_per_day[100000]:<15.2f} ")
