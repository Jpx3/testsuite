from copy import deepcopy

import numpy
import numpy as np
import torch
from ase.db.core import Database
from e3nn.o3 import angles_to_matrix
from tqdm import tqdm

from data.augment import rotate_in_place
from data.loaders import MyAseSqliteDataset


def do_some_stuff(driver, model_variation, dataset: MyAseSqliteDataset, reuse_calc=True):
  torch.set_printoptions(precision=4, sci_mode=False)
  torch.random.manual_seed(16)
  numpy.random.seed(16)
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  torch.autograd.set_detect_anomaly(True)
  calc = None

  if reuse_calc:
    calc = driver.get_ase_calculator(model_variation)

  for data in dataset:
    atoms = deepcopy(data)
    n_a = 8
    n_b = 8

    linspace_a = torch.linspace(0, 2 * torch.pi, steps=n_a, device=device)
    linspace_b = torch.linspace(0, 2 * torch.pi, steps=n_b, device=device)
    errors = torch.zeros((n_a, n_b), device=device, dtype=torch.float64)
    # Do not change
    cnt_a = 0
    cnt_b = 0

    last_n_node = torch.tensor([], dtype=torch.int64)
    last_n_edge = torch.tensor([], dtype=torch.int64)

    energies = torch.zeros((n_a, n_b), device=device, dtype=torch.float64)

    for rot_a in tqdm(linspace_a):
      for rot_b in linspace_b:
        rot_matrix_3d = angles_to_matrix(rot_a, rot_b, torch.tensor(0, device=device)).to(device)

        if np.abs(np.linalg.det(rot_matrix_3d.cpu().numpy()) - 1) > 1e-6:
          print(f"Rotation matrix corrupt: {rot_matrix_3d.cpu().numpy()}")

        rotate_in_place(atoms, rot_matrix_3d.cpu().numpy())

        if not reuse_calc:
          calc = driver.get_ase_calculator(model_variation)

        atoms.calc = calc
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
        energies[cnt_a, cnt_b] = energy

        cnt_b += 1
      cnt_b = 0
      cnt_a += 1

    print("Energies stats:", energies.min().item(), energies.max().item(), energies.mean().item(), energies.std().item())
