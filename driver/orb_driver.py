from ase.calculators.calculator import Calculator
from orb_models.forcefield import pretrained
from orb_models.forcefield.calculator import ORBCalculator

from driver.driver import Driver
import torch


class Orbv3Driver(Driver):

  def get_model_variations(self):
    X = ["conservative", "direct"]
    Y = ["inf", "20"]
    Z = ["omat", "mptrj"]
    model_variations = set()
    for x in X:
      for y in Y:
        for z in Z:
          model_variations.add(f"{x}_{y}_{z}")
    return model_variations

  def get_ase_calculator(
    self, model_variation, device=torch.get_default_device(),
    compile=True, **kwargs
  ) -> Calculator:
    compile = False
    if model_variation == "conservative_inf_omat":
      model = pretrained.orb_v3_conservative_inf_omat(compile=compile, device=device)
    elif model_variation == "conservative_20_omat":
      model = pretrained.orb_v3_conservative_20_omat(compile=compile, device=device)
    elif model_variation == "conservative_inf_mptrj":
      model = pretrained.orb_v3_conservative_inf_mpa(compile=compile, device=device)
    elif model_variation == "conservative_20_mptrj":
      model = pretrained.orb_v3_conservative_20_mpa(compile=compile, device=device)
    elif model_variation == "direct_inf_omat":
      model = pretrained.orb_v3_direct_inf_omat(compile=compile, device=device)
    elif model_variation == "direct_20_omat":
      model = pretrained.orb_v3_direct_20_omat(compile=compile, device=device)
    elif model_variation == "direct_inf_mptrj":
      model = pretrained.orb_v3_direct_inf_mpa(compile=compile, device=device)
    elif model_variation == "direct_20_mptrj":
      model = pretrained.orb_v3_direct_20_mpa(compile=compile, device=device)
    else:
      raise ValueError(f"Unknown model variation: {model_variation}")
    calc = ORBCalculator(model=model, device=device)
    return calc

  def supports_compilation_option(self) -> bool:
    return True

  def is_conservative(self, model_variation) -> bool:
    return model_variation.startswith("conservative")


class Orbv2Driver(Driver):

  def get_model_variations(self):
    Y = ["normal", "mptrj-only"]
    model_variations = set()
    for y in Y:
      model_variations.add(f"direct_{y}")

  def get_ase_calculator(
    self, model_variation, device=torch.get_default_device(),
    compile=False, **kwargs
  ) -> Calculator:
    compile = False
    if model_variation == "direct_normal":
      model = pretrained.orb_v2(compile=compile, device=device)
    elif model_variation == "direct_mptrj-only":
      model = pretrained.orb_mptraj_only_v2(compile=compile, device=device)
    else:
      raise ValueError(f"Unknown model variation: {model_variation}")
    calc = ORBCalculator(model=model, device=device)
    return calc

  def is_conservative(self, model_variation) -> bool:
    return model_variation.startswith("conservative")