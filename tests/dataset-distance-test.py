import torch
from ase.calculators.lj import LennardJones
from ase.md import VelocityVerlet
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from orb_models.forcefield.calculator import ORBCalculator
from scipy.constants import Boltzmann
from tqdm import tqdm

from data.mptraj import mptraj_testset
import numpy as np

from driver.gpaw_driver import GPAWDFTDriver
from driver.orb_driver import Orbv3Driver

mptraj = mptraj_testset("../data/mptrj-norm-test.db")

if __name__ == "__main__":
  distances = []
  # forces = []

  atom_mean_dists = []
  force_dists = []

  # driver = Orbv3Driver(name="orb")
  # calc = driver.get_ase_calculator("direct_inf_omat")
  driver = GPAWDFTDriver(name="gpaw")
  calc = driver.get_ase_calculator(model_variation="lcao")

  i = 5000
  for atoms in tqdm(mptraj):
    # Compute pairwise distance of the atoms
    rij = atoms.get_all_distances(mic=True)
    rij[rij == 0.0] = torch.nan
    atoms.calc = calc

    # # Heat
    # MaxwellBoltzmannDistribution(atoms, temperature_K=300)
    # # Perform some MD
    # VelocityVerlet(atoms, 1.0).run(10)

    atom_mean_dist = np.nanmin(rij, axis=1)
    forces = atoms.get_forces()
    forces = np.linalg.norm(forces, axis=1)
    atom_mean_dists.append(atom_mean_dist)
    force_dists.append(forces)

    rij[rij > 8.0] = torch.nan
    distances.append(rij[rij > 0.0])
    i = i - 1
    if i <= 0:
      break
    # break

  # Histogram of distances
  import matplotlib.pyplot as plt
  import numpy as np
  distances = np.concatenate(distances)
  plt.hist(distances, bins=200, range=(0, 8.0))
  plt.yscale("log")
  plt.xlabel("Distance (Å)")
  plt.ylabel("Count")
  plt.title("Histogram of atomic distances in MPtrj")
  # plt.savefig("distance-histogram.png", dpi=300)
  plt.show()

  # Scatter plot of mean atomic distance vs force magnitude
  atom_mean_dists = np.concatenate(atom_mean_dists)
  force_dists = np.concatenate(force_dists)

  # Merge into a 2D histogram
  plt.hist2d(atom_mean_dists, force_dists, bins=100, range
              =[[0, 8.0], [0, 2.0]], cmin=1)
  plt.colorbar(label="Count")
  plt.xlabel("Min atomic distance of Atom (Å)")
  plt.ylabel("Force magnitude (eV/Å)")
  plt.title("Min atomic distance vs Force magnitude in MPtrj")
  # plt.savefig("distance-vs-force.png", dpi=300)
  plt.show()