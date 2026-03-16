import pickle

import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms
from tqdm import tqdm

element_names = {
  "H": "Hydrogen",
  "Li": "Lithium",
  "Be": "Beryllium",
  "B": "Boron",
  "C": "Carbon",
  "N": "Nitrogen",
  "O": "Oxygen",
  "F": "Fluorine",
  "Na": "Sodium",
  "Mg": "Magnesium",
  "Al": "Aluminum",
  "Si": "Silicon",
  "P": "Phosphorus",
  "S": "Sulfur",
  "Cl": "Chlorine",
  "K": "Potassium",
  "Ca": "Calcium",
  "Sc": "Scandium",
  "Ti": "Titanium",
  "V": "Vanadium",
  "Cr": "Chromium",
  "Mn": "Manganese",
  "Fe": "Iron",
  "Co": "Cobalt",
  "Ni": "Nickel",
  "Cu": "Copper",
  "Zn": "Zinc",
  "Ga": "Gallium",
  "Ge": "Germanium",
  "As": "Arsenic",
  "Se": "Selenium",
  "Br": "Bromine",
}


def get_element_name(symbol: str) -> str:
  normalized = symbol.capitalize()
  return element_names.get(normalized, f"Unknown symbol: {symbol}")


if __name__ == "__main__":
  # models
  # curves = pickle.load(open("diatomic_curves.pkl", "rb"))
  folder = "../results/diatomic_energy_curve/"
  import os

  files = os.listdir(folder)

  elements = set()

  first_file = next(iter(files))
  if first_file.endswith(".pkl"):
    dat = pickle.load(open(os.path.join(folder, first_file), "rb"))
    for element in dat.keys():
      elements.add(element)

  files = os.listdir(folder)

  for file in files:
    if file.endswith(".pkl"):
      dat = pickle.load(open(os.path.join(folder, file), "rb"))
      # Remove elements if they are not in this file
      for element in list(elements):
        if element not in dat.keys():
          print(f"Removing element {element} as it is not in file {file}")
          elements.remove(element)

  print(f"Common elements across all files: {elements}")

  for element in elements:
    plt.figure(figsize=(8, 6))  # width, height in inches

    lowest_energy = 100
    files = os.listdir(folder)
    for file in files:
      if file.endswith(".pkl"):
        name = file.replace(".pkl", "")
        dat = pickle.load(open(os.path.join(folder, file), "rb"))
        rotation_curves = dat[element]

        if name.lower().startswith("dft_gpaw_lcao"):
          continue

        is_dft = name.lower().startswith("dft")
        n_rots = len(rotation_curves)
        n_dist = len(rotation_curves[0]["distances"])

        data = np.zeros((n_rots, n_dist))
        for i, curve in enumerate(rotation_curves):
          data[i, :] = curve["energies"]
          if np.min(curve["energies"]) < lowest_energy:
            lowest_energy = np.min(curve["energies"])
            # print(f"New lowest energy found: {lowest_energy} eV for {element} in {name}")
        # Ignore data points == 0 for dft and replace with nan
        if is_dft:
          data[data == 0] = np.nan

        mean_curve = np.mean(data, axis=0)
        std_curve = np.std(data, axis=0)
        bottom_95 = np.percentile(data, 2.5, axis=0)
        top_95 = np.percentile(data, 97.5, axis=0)
        distances = rotation_curves[0]["distances"]

        if name.lower() == "orb_v3_direct_inf_omat":
          # Print element name and energy at 6.0 Å
          energy_at_6A = mean_curve[np.argmin(np.abs(distances - 6.0))]
          print(f"Element: {element}, Model: {name}, Energy at 6.0 Å: {energy_at_6A} eV")


        from model_colors import model_colors
        color = model_colors.get(name, None)

        # Gradient descent on the mean curve
        def energy_func(r):
          return np.interp(r, distances, mean_curve)
        def energy_grad(r):
          h = 1e-5
          return (energy_func(r + h) - energy_func(r - h)) / (2 * h)
        initial_distances = np.arange(0.5, 6.1, 0.1)
        min_energies = []
        local_minima_positions = []
        for r0 in initial_distances:
          r = r0
          lr, v, m = 0.01, 0, 0
          beta1, beta2, eps = 0.8, 0.8, 1e-8
          pos_eps = 5
          for step in range(1000):
            grad = energy_grad(r)
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * (grad ** 2)
            m_hat = m / (1 - beta1 ** (step + 1))
            v_hat = v / (1 - beta2 ** (step + 1))
            chng = lr * m_hat / (np.sqrt(v_hat) + eps)
            r -= chng
            r = max(r, 0.1)
            r = min(r, 6.0)
            pos_eps = 0.9 * pos_eps + 0.1 * abs(chng)
            # print(pos_eps)
            if pos_eps < 0.003:
              break
            # print(f"Step {step}: r = {r:.4f}, grad = {grad:.6f}, m = {m_hat:.6f}, v = {v_hat:.6f}")
            # if abs(grad) < 1e-5:
            #   break
          local_minima_positions.append(r)
          min_energies.append(energy_func(r))

        # Plot minimized energy curve
        min_energies = np.array(min_energies)

        # DFT is black crosses
        if is_dft:
          # Must be the first plot to be on top
          plt.plot(
            distances, mean_curve,
            label=f"{name}", color='black',
            # marker='x', linestyle='None'
            linestyle='--', marker='o', markersize=4,
            zorder=10
          )
        else:
          # Make thicker lines
          plt.plot(distances, mean_curve, label=f"{name}", color=color, linewidth=2)

          positions_drawn = set()
          # Plot unique minimized energies as points
          for pos, energy in zip(local_minima_positions, min_energies):
            # To int
            pos_rounded = round(pos, 1)
            if pos_rounded in positions_drawn:
              continue
            if energy > 50:
              continue
            plt.scatter(pos, energy, color=color, s=30, zorder=5, marker='D', alpha=0.35)
            # plt.text(pos, energy + 0.5, f"{energy:.2f} eV", color=color, fontsize=8, alpha=0.7)
            positions_drawn.add(pos_rounded)


        plt.fill_between(distances, mean_curve - std_curve, mean_curve + std_curve,
                         alpha=0.4, color=color)
        # plt.fill_between(distances, bottom_95, top_95, alpha=0.1)

    plt.xlabel("Interatomic Distance (Å)")
    plt.ylabel("Potential Energy (eV)")
    plt.title("Diatomic Energy Curves for " + get_element_name(element))
    # upper limit 50 eV, lower limit lowest_energy - 1 eV
    plt.ylim(lowest_energy - 1, 50)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper right')
    plt.tight_layout()
    plt.grid()
    # Create figures/diatomic if it does not exist
    if not os.path.exists("figures/diatomic"):
      os.makedirs("figures/diatomic")

    plt.savefig(f"figures/diatomic/{get_element_name(element).lower()}.png", dpi=300)
    plt.show()

  # output_folder = "figures/diatomic_energy_min"
  # os.makedirs(output_folder, exist_ok=True)
  #
  # for element in elements:
  #   plt.figure(figsize=(10, 7))  # wider for multiple lines
  #
  #   lowest_energy = np.inf
  #   files = os.listdir(folder)
  #
  #   for file in tqdm(files):
  #     if not file.endswith(".pkl"):
  #       continue
  #
  #     name = file.replace(".pkl", "")
  #     dat = pickle.load(open(os.path.join(folder, file), "rb"))
  #     rotation_curves = dat[element]
  #
  #     n_rots = len(rotation_curves)
  #     n_dist = len(rotation_curves[0]["distances"])
  #
  #     # Collect all energies for averaging
  #     data = np.zeros((n_rots, n_dist))
  #     for i, curve in enumerate(rotation_curves):
  #       data[i, :] = curve["energies"]
  #       min_curve_energy = np.min(curve["energies"])
  #       if min_curve_energy < lowest_energy:
  #         lowest_energy = min_curve_energy
  #         print(f"New lowest energy found: {lowest_energy:.3f} eV for {element} in {name}")
  #
  #     mean_curve = np.mean(data, axis=0)
  #     std_curve = np.std(data, axis=0)
  #     distances = rotation_curves[0]["distances"]
  #
  #     color = model_colors.get(name, "gray")
  #
  #     # Plot mean ± std of original rotation curves
  #     plt.fill_between(distances, mean_curve - std_curve, mean_curve + std_curve, color=color, alpha=0.2)
  #     plt.plot(distances, mean_curve, color=color, linewidth=2, label=f"{name} (mean)")
  #
  #
  #     # Gradient descent on the mean curve
  #     def energy_func(r):
  #       return np.interp(r, distances, mean_curve)
  #
  #
  #     def energy_grad(r):
  #       h = 1e-5
  #       return (energy_func(r + h) - energy_func(r - h)) / (2 * h)
  #
  #
  #     initial_distances = np.arange(0.5, 6.1, 0.1)
  #     min_energies = []
  #
  #     for r0 in initial_distances:
  #       r = r0
  #       lr, v, m = 0.01, 0, 0
  #       beta1, beta2, eps = 0.9, 0.999, 1e-8
  #       for step in range(1000):
  #         grad = energy_grad(r)
  #         m = beta1 * m + (1 - beta1) * grad
  #         v = beta2 * v + (1 - beta2) * (grad ** 2)
  #         m_hat = m / (1 - beta1 ** (step + 1))
  #         v_hat = v / (1 - beta2 ** (step + 1))
  #         r -= lr * m_hat / (np.sqrt(v_hat) + eps)
  #         r = max(r, 0.1)
  #         r = min(r, 6.0)
  #         if abs(grad) < 1e-5:
  #           break
  #       min_energies.append(energy_func(r))
  #
  #     # Plot minimized energy curve
  #     min_energies = np.array(min_energies)
  #     plt.plot(initial_distances, min_energies, linestyle='--', color=color, linewidth=2)
  #     min_idx = np.argmin(min_energies)
  #     plt.scatter(initial_distances[min_idx], min_energies[min_idx], color=color, s=50, zorder=5)
  #     plt.text(initial_distances[min_idx], min_energies[min_idx] + 0.5,
  #              f"{min_energies[min_idx]:.2f} eV", color=color)
  #
  #   plt.xlabel("Interatomic Distance (Å)", fontsize=14)
  #   plt.ylabel("Energy (eV)", fontsize=14)
  #   plt.title(f"Minimized Diatomic Energies for {get_element_name(element)}", fontsize=16)
  #   plt.ylim(lowest_energy - 1, 50)
  #   plt.grid(alpha=0.3)
  #   plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
  #   plt.tight_layout()
  #
  #   save_path = os.path.join(output_folder, f"{get_element_name(element).lower()}.png")
  #   plt.savefig(save_path, dpi=300)
  #   plt.close()
  #

