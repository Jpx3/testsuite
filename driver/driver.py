from abc import ABC, abstractmethod

import ase
import torch
from ase.calculators.calculator import Calculator

class Driver(ABC):
  def __init__(self, name):
    self.name = name

  @abstractmethod
  def get_model_variations(self):
    pass

  @abstractmethod
  def get_ase_calculator(self, model_variation, device=torch.get_default_device(), compile=True, **kwargs) -> Calculator:
    pass

  def supports_reuse(self, atoms: ase.Atoms) -> bool:
    return True

  def supports_compilation_option(self) -> bool:
    return False

  def supports_system_specific_compilation(self) -> bool:
    return False

  @abstractmethod
  def is_conservative(self, model_variation) -> bool:
    pass

  def __str__(self):
    return f"Driver: {self.name}"
