from driver.driver import Driver
from ase.calculators.calculator import Calculator


from nequip.ase import NequIPCalculator

import torch

class NequipDriver(Driver):

  def get_model_variations(self):
    return ["default"]

  def get_ase_calculator(
    self, model_variation, device=torch.get_default_device(),
    compile=True, **kwargs
  ) -> Calculator:
    return NequIPCalculator.from_compiled_model(
      compile_path="models/mir-group__Allegro-MP-L__0.1.nequip.pt2",
      device=device,
    )

  def is_conservative(self, model_variation) -> bool:
    return True
