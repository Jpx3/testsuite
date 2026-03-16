from mattersim.forcefield import MatterSimCalculator

from driver.driver import Driver
from ase.calculators.calculator import Calculator

import torch

class MatterSimDriver(Driver):

  def get_model_variations(self):
    return ["v1"]

  def get_ase_calculator(
    self, model_variation,
    device=torch.get_default_device(), compile=True, **kwargs
  ) -> Calculator:
    if model_variation not in self.get_model_variations():
      raise ValueError(f"Unsupported model variation: {model_variation}, supported: {self.get_model_variations()}")
    device = torch.device(device)
    is_cuda = device.type == 'cuda'
    return MatterSimCalculator(
      device="cuda" if is_cuda else "cpu",
    )

  def is_conservative(self, model_variation) -> bool:
    return True
