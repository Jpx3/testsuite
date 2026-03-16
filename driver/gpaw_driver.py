import torch
from ase.calculators.calculator import Calculator
from ase.calculators.emt import EMT

from driver.driver import Driver


class GPAWDFTDriver(Driver):
  def get_model_variations(self):
    return ["lcao", "pw", "fd"]

  def get_ase_calculator(
    self,
    model_variation="pw",
    device=torch.get_default_device(),
    compile=True,
    **kwargs
  ) -> Calculator:
    # from gpaw import GPAW
    # calc = GPAW(mode=model_variation, basis="dzp", xc="PBE", txt="gpaw.txt")
    return EMT()

  def is_conservative(self, model_variation) -> bool:
    return True
