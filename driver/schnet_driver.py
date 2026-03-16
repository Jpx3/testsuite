import importlib

from fairchem.core import OCPCalculator
# from fairchem.core.calculate.pretrained_mlip import pretrained_checkpoint_path_from_name
from fairchem.core.models import model_registry
# from fairchem.core.units.mlip_unit import load_predict_unit

from driver.driver import Driver
from ase.calculators.calculator import Calculator

# from fairchem.core import pretrained_mlip, OCPCalculator
# from fairchem.core import FAIRChemCalculator

import torch

class SchNetDriver(Driver):

  def get_model_variations(self):
    return ["oc20", "odac"]

  def get_ase_calculator(
    self, model_variation,
    device=torch.get_default_device(),
    compile=True, **kwargs
  ) -> Calculator:
    # Available: ('CGCNN-S2EF-OC20-200k', 'CGCNN-S2EF-OC20-2M',
    # 'CGCNN-S2EF-OC20-20M', 'CGCNN-S2EF-OC20-All',
    # 'DimeNet-S2EF-OC20-200k', 'DimeNet-S2EF-OC20-2M',
    # 'SchNet-S2EF-OC20-200k', 'SchNet-S2EF-OC20-2M',
    # 'SchNet-S2EF-OC20-20M', 'SchNet-S2EF-OC20-All',
    # 'DimeNet++-S2EF-OC20-200k', 'DimeNet++-S2EF-OC20-2M',
    # 'DimeNet++-S2EF-OC20-20M', 'DimeNet++-S2EF-OC20-All',
    # 'SpinConv-S2EF-OC20-2M', 'SpinConv-S2EF-OC20-All',
    # 'GemNet-dT-S2EF-OC20-2M', 'GemNet-dT-S2EF-OC20-All',
    # 'PaiNN-S2EF-OC20-All', 'GemNet-OC-S2EF-OC20-2M',
    # 'GemNet-OC-S2EF-OC20-All', 'GemNet-OC-S2EF-OC20-All+MD',
    # 'GemNet-OC-Large-S2EF-OC20-All+MD', 'SCN-S2EF-OC20-2M',
    # 'SCN-t4-b2-S2EF-OC20-2M', 'SCN-S2EF-OC20-All+MD',
    # 'eSCN-L4-M2-Lay12-S2EF-OC20-2M',
    # 'eSCN-L6-M2-Lay12-S2EF-OC20-2M',
    # 'eSCN-L6-M2-Lay12-S2EF-OC20-All+MD',
    # 'eSCN-L6-M3-Lay20-S2EF-OC20-All+MD',
    # 'EquiformerV2-83M-S2EF-OC20-2M',
    # 'EquiformerV2-31M-S2EF-OC20-All+MD',
    # 'EquiformerV2-153M-S2EF-OC20-All+MD',
    # 'SchNet-S2EF-force-only-OC20-All',
    # 'DimeNet++-force-only-OC20-All',
    # 'DimeNet++-Large-S2EF-force-only-OC20-All',
    # 'DimeNet++-S2EF-force-only-OC20-20M+Rattled',
    # 'DimeNet++-S2EF-force-only-OC20-20M+MD',
    # 'CGCNN-IS2RE-OC20-10k', 'CGCNN-IS2RE-OC20-100k',
    # 'CGCNN-IS2RE-OC20-All', 'DimeNet-IS2RE-OC20-10k',
    # 'DimeNet-IS2RE-OC20-100k', 'DimeNet-IS2RE-OC20-all',
    # 'SchNet-IS2RE-OC20-10k', 'SchNet-IS2RE-OC20-100k',
    # 'SchNet-IS2RE-OC20-All', 'DimeNet++-IS2RE-OC20-10k',
    # 'DimeNet++-IS2RE-OC20-100k', 'DimeNet++-IS2RE-OC20-All',
    # 'PaiNN-IS2RE-OC20-All', 'GemNet-dT-S2EFS-OC22',
    # 'GemNet-OC-S2EFS-OC22', 'GemNet-OC-S2EFS-OC20+OC22',
    # 'GemNet-OC-S2EFS-nsn-OC20+OC22',
    # 'GemNet-OC-S2EFS-OC20->OC22',
    # 'EquiformerV2-lE4-lF100-S2EFS-OC22',
    # 'SchNet-S2EF-ODAC', 'DimeNet++-S2EF-ODAC',
    # 'PaiNN-S2EF-ODAC', 'GemNet-OC-S2EF-ODAC',
    # 'eSCN-S2EF-ODAC', 'EquiformerV2-S2EF-ODAC',
    # 'EquiformerV2-Large-S2EF-ODAC', 'Gemnet-OC-IS2RE-ODAC',
    # 'eSCN-IS2RE-ODAC', 'EquiformerV2-IS2RE-ODAC',
    # 'EquiformerV2-153M-OMAT24',
    # 'EquiformerV2-153M-OMAT24-MP-sAlex',
    # 'EquiformerV2-31M-MP', 'EquiformerV2-31M-OMAT24',
    # 'EquiformerV2-31M-OMAT24-MP-sAlex', 'EquiformerV2-86M-OMAT24',
    # 'EquiformerV2-86M-OMAT24-MP-sAlex',
    # 'EquiformerV2-DeNS-153M-MP', 'EquiformerV2-DeNS-86M-MP',
    # 'EquiformerV2-DeNS-31M-MP', 'eSEN-30M-OMAT24',
    # 'eSEN-30M-OAM', 'eSEN-30M-MP'
    # )

    type = model_variation.lower()
    device = torch.device(device)
    # print(f"Available: {model_registry.available_pretrained_models}")
    # create cache directory
    import os
    local_cache = os.path.expanduser("~/.fairchem/models/")
    if not os.path.exists(local_cache):
      os.makedirs(local_cache)

    model_name = ""
    if type == "oc20":
      model_name = "SchNet-S2EF-OC20-All"
    elif type == "odac":
      model_name = "SchNet-S2EF-ODAC"
    calc = OCPCalculator(
      model_name=model_name, local_cache=local_cache,
      cpu=device.type == "cpu",
    )
    return calc

  def is_conservative(self, model_variation) -> bool:
    return True
