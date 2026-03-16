from tqdm import tqdm

from driver.driver import Driver
import numpy as np
from ase import Atoms

from ase.calculators.calculator import Calculator

def distance_test(
  driver: Driver,
  model_variation: str,
):
  device = "cuda"
  calc = driver.get_ase_calculator(model_variation=model_variation, device=device)

  radius_start = 0.5
  radius_end = 6.0
  n_r = 50
  distances = np.linspace(radius_start, radius_end, n_r)

  n_a, n_b, n_c = 3, 3, 3
  if driver.name.startswith("dft"):
    n_a, n_b, n_c = 1, 1, 1

  rotations = generate_rotations(n_a, n_b, n_c)

  elements = available_elements()

  print(f"Testing {len(elements)} elements with {len(rotations)} rotations each.")

  all_results = dict()
  for element in elements:
    print("Testing", element)
    element_results = []
    for i, R in enumerate(tqdm(rotations)):
      result = single_distance_test(
        calc=calc,
        element=element,
        distances=distances,
        rotation_matrix=R,
      )
      element_results.append(result)
    all_results[element] = element_results

  return all_results


def single_distance_test(
  calc: Calculator,
  element: str = "Al",
  distances: np.ndarray = None,
  rotation_matrix: np.ndarray = None,
):
  results = dict()
  if distances is None:
    distances = np.linspace(0.35, 6.0, 20)
  if rotation_matrix is None:
    rotation_matrix = np.eye(3)

  energies = []
  for d in distances:
    cell = np.array([50, 50, 50])
    midpoint = cell / 2
    positions = np.array([[0, 0, 0], rotation_matrix @ [d, 0, 0]]) + midpoint
    mol = Atoms(f"{element}2", positions=positions, cell=cell, pbc=[0, 0, 0])
    mol.calc = calc
    try:
      energy = mol.get_potential_energy()
      # print(f"Distance: {d:.2f}, Energy: {energy:.4f}")
    except Exception as e:
      print(f"Error for distance {d} and element {element}: {e}")
      energy = 0
    energies.append(energy)

  results["distances"] = distances
  results["energies"] = np.array(energies)
  return results


def available_elements(
  include_noble_gases: bool = False,
):
  elements = [
    "H", "Li", "Be", "B", "C", "N", "O", "F", "Na", "Mg", "Al", "Si", "P", "S", "Cl",
    "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br",
  ]
  noble_gases = ["He", "Ne", "Ar", "Kr", "Xe", "Rn", "Og"]
  if include_noble_gases:
    elements += noble_gases
  return elements


def generate_rotations(
  n_a, n_b, n_c
):
  rotations = []
  angles_a = np.linspace(0, 2 * np.pi, n_a, endpoint=False)
  angles_b = np.linspace(0, 2 * np.pi, n_b, endpoint=False)
  angles_c = np.linspace(0, 2 * np.pi, n_c, endpoint=False)
  for alpha in angles_a:
    R_x = np.array([
      [1, 0, 0],
      [0, np.cos(alpha), -np.sin(alpha)],
      [0, np.sin(alpha), np.cos(alpha)]
    ])
    for beta in angles_b:
      R_y = np.array([
        [np.cos(beta), 0, np.sin(beta)],
        [0, 1, 0],
        [-np.sin(beta), 0, np.cos(beta)]
      ])
      for gamma in angles_c:
        R_z = np.array([
          [np.cos(gamma), -np.sin(gamma), 0],
          [np.sin(gamma), np.cos(gamma), 0],
          [0, 0, 1]
        ])
        rotations.append(R_z @ R_y @ R_x)

  for rot in rotations:
    det = np.linalg.det(rot)
    assert np.isclose(det, 1.0), f"Rotation matrix determinant is not 1: {det}"
  return rotations
