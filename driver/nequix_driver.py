from driver.driver import Driver
from ase.calculators.calculator import Calculator

from nequix.calculator import NequixCalculator

import torch

class NequixDriver(Driver):

  def get_model_variations(self):
    return ["default"]

  def get_ase_calculator(
    self, model_variation, device=torch.get_default_device(),
    compile=True, **kwargs
  ) -> Calculator:
    device = torch.device(device)
    from jax import config
    if device.type == "cpu":
      config.update("jax_platform_name", "cpu")
    elif device.type == "cuda":
      config.update("jax_platform_name", "gpu")
    return NequixCalculator(device=device)

  def is_conservative(self, model_variation) -> bool:
    return True
