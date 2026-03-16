import ase
import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.filters import FrechetCellFilter
from ase.md import VelocityVerlet
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.optimize import FIRE
from ase import units

from data.mptraj import mptraj_testset
from driver.orb_driver import Orbv3Driver

mptraj = mptraj_testset("../data/mptrj-norm-test.db")

def test_relax(
  atoms: Atoms,
  calc: Calculator
):
  copy_atoms = atoms.copy()
  atoms.set_calculator(calc)

  # Deterministic thermostat
  MaxwellBoltzmannDistribution(
    atoms,
    temperature_K=300,
    rng=np.random.RandomState(1337)
  )

  dyn = VelocityVerlet(atoms, dt=units.fs * 1)
  dyn.run(steps=10)

  dyn = FIRE(atoms, dt=units.fs * 1)
  dyn.run(fmax=0.01, steps=200)




if __name__ == "__main__":
  driver = Orbv3Driver(name="orb")
  calc = driver.get_ase_calculator("conservative_inf_omat")
  first_atom = mptraj[0]
  test_relax(first_atom, calc)
