from ase.calculators.calculator import Calculator

from driver.driver import Driver
import torch

from mace.calculators import mace_mp

class MACEDriver(Driver):

  def get_model_variations(self):
    model_variations = {"large"}
    return model_variations

  def get_ase_calculator(
    self, model_variation, device=torch.get_default_device(),
    compile=True, **kwargs
  ) -> Calculator:
    device = torch.device(device)
    patch()

    calc = mace_mp(
      # model="medium",
      device="cuda" if device.type == "cuda" else "cpu",
      dispersion=True,
      compile=compile,
    )
    return calc

  def supports_compilation_option(self) -> bool:
    return True

  def is_conservative(self, model_variation) -> bool:
    return False


def patch():
  from e3nn.util.codegen import _mixin

  _old_setstate = _mixin.CodeGenMixin.__setstate__

  def patched_setstate(self, codegen_state):
    try:
      # If it's a dict, try normal unpacking
      if isinstance(codegen_state, dict):
        _old_setstate(self, codegen_state)
      else:
        # If it's anything else (bool, None, etc.), skip it safely
        _old_setstate(self, {})
    except Exception:
      # If the first attempt failed, fallback: only handle dicts
      if isinstance(codegen_state, dict):
        # Ensure values are tuples for old-style codegen dicts
        new_state = {}
        for k, v in codegen_state.items():
          try:
            new_state[k] = tuple(v)
          except Exception:
            new_state[k] = v  # keep original if not iterable
        _old_setstate(self, new_state)
      else:
        # ultimate fallback: pass empty dict
        _old_setstate(self, {})

  _mixin.CodeGenMixin.__setstate__ = patched_setstate
