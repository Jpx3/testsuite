from sevenn.calculator import SevenNetCalculator

from driver.driver import Driver
from ase.calculators.calculator import Calculator

import torch

class SevenNetDriver(Driver):

  def get_model_variations(self):
    return ["7net-mf-ompa"]

  def get_ase_calculator(
    self, model_variation, device=torch.get_default_device(),
    compile=True, **kwargs
  ) -> Calculator:
    if model_variation not in self.get_model_variations():
      raise ValueError(f"Unsupported model variation: {model_variation}, supported: {self.get_model_variations()}")
    return SevenNetCalculator(
      model_variation, modal='mpa',
      device=device
    )

  def is_conservative(self, model_variation) -> bool:
    return True
