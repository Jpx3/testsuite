from fairchem.core.units.mlip_unit import InferenceSettings

from driver.driver import Driver
from ase.calculators.calculator import Calculator

from fairchem.core import pretrained_mlip
from fairchem.core import FAIRChemCalculator

import torch

class UMADriver(Driver):

  def get_model_variations(self):
    X = ["s-1p1", "m-1p1"]
    Y = ["oc20", "omat", "omol", "odac", "omc"]
    model_variations = set()
    for x in X:
      for y in Y:
        model_variations.add(f"{x}_{y}")
    return model_variations

  def get_ase_calculator(
    self, model_variation, device=torch.get_default_device(),
    compile=False, **kwargs
  ) -> Calculator:
    x = model_variation.split("_")[0]
    y = model_variation.split("_")[1]

    # If x doesn't start with "uma-", prepend "uma-"
    if not x.startswith("uma-"):
      x = "uma-" + x

    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

    device = torch.device(device)
    device_is_cuda = device.type == "cuda"

    settings = InferenceSettings(
      tf32=True,
      activation_checkpointing=False,
      merge_mole=kwargs.get("syscache", False),
      compile=False,  # compilation doesn't seem to work yet
      wigner_cuda=False,
      external_graph_gen=False,
      internal_graph_gen_version=2,
    )

    model = pretrained_mlip.get_predict_unit(
      x, device="cuda" if device_is_cuda else "cpu",
      inference_settings=settings,
    )
    calc = FAIRChemCalculator(predict_unit=model, task_name=y)
    return calc

  def is_conservative(self, model_variation) -> bool:
    return model_variation.startswith("conservative")
