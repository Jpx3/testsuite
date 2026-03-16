from typing import Callable

import pickle

import driver.drivers
from data.mptraj import mptraj_testset
from driver.driver import Driver
from tests import phonon_bandstructure, diatomic_energy_curve, inference_times, energydrift, co2_stability, mof_widom, \
  rotational_effects


def start():
  import matplotlib
  matplotlib.use("Agg")

  # Check if torch is available
  try:
    import torch
    print(f"Using torch version {torch.__version__}")
    print(f"Torch CUDA is available: {torch.cuda.is_available()}")
  except ImportError:
    print("Torch is not installed. Please install torch to run the tests.")
    return

  orb_v3 = driver.drivers.Orbv3Driver("orb_v3")
  orb_v2 = driver.drivers.Orbv2Driver("orb_v2")
  # uma = driver.drivers.UMADriver("uma")
  # nequip = driver.drivers.NequipDriver("nequip")
  nequix = driver.drivers.NequixDriver("nequix")
  # mattersim = driver.drivers.MatterSimDriver("mattersim")
  sevennet = driver.drivers.SevenNetDriver("sevennet")
  # gpaw = driver.drivers.GPAWDFTDriver("dft_gpaw")
  eSEN = driver.drivers.eSENDriver("eSEN")
  painn = driver.drivers.PaiNNDriver("painn")
  schnet = driver.drivers.SchNetDriver("schnet")

  variation = "conservative_inf_omat"
  direct_variation = "direct_inf_omat"
  v2_variation = "direct_mptrj-only"
  # uma_variation = "m-1p1_omat"
  # nequip_variation = "default"
  nequix_variation = "default"
  # mattersim_variation = "v1"
  sevennet_variation = "7net-mf-ompa"
  # gpaw_variation = "pw"
  eSEN_variation = "oam"
  painn_variation = "oc20"
  schnet_variation = "oc20"

  drivers = [
    orb_v2, orb_v3, orb_v3,
    # uma,
    nequix,
    # mattersim,
    eSEN,
    sevennet,
    painn,
    schnet,
  ]
  variations = [
    v2_variation, variation, direct_variation,
    # uma_variation,
    nequix_variation,
    # mattersim_variation,
    eSEN_variation,
    sevennet_variation,
    painn_variation,
    schnet_variation,
  ]

  for d, v in zip(drivers, variations):
    print(f"Running tests for {d.name} with variation {v}")
    run_test(
      phonon_bandstructure.full_phonon_test,
      test_name="phonon_norelax",
      driver=d, model_variation=v,
    )
    run_test(
      inference_times.test_inference_time,
      test_name="inference_time",
      driver=d, model_variation=v,
    )
    run_test(
      runner=diatomic_energy_curve.distance_test,
      test_name="diatomic_energy_curve",
      driver=d, model_variation=v,
    )
    run_test(
      runner=co2_stability.test_co2_stability,
      test_name="co2_stability",
      driver=d, model_variation=v,
    )
    run_test(
      runner=mof_widom.test_co2_insertion_to_mof,
      test_name="mof_co2_insertion",
      driver=d, model_variation=v,
    )
    # run_test(
    #   runner=rotational_effects.test_rotational_effects,
    #   test_name="rotational_effects",
    #   driver=d, model_variation=v,
    # )

def run_test(
  runner: Callable[[Driver, str], dict],
  test_name: str,
  driver: Driver,
  model_variation: str,
) -> dict:
  # Check for existing results
  import os

  # results/{test_name}/{driver.name}_{model_variation}.pkl
  results_file = f"results/{test_name}/{driver.name}_{model_variation}.pkl"

  # Make sure all folders exist
  os.makedirs(os.path.dirname(results_file), exist_ok=True)

  if os.path.exists(results_file):
    with open(results_file, "rb") as f:
      results = pickle.load(f)
      print(f"  Found existing results from {results_file}")
      return results
  else:
    print(f"  Running {test_name} for {driver.name} with variation {model_variation}")
    results = runner(driver, model_variation)
    with open(results_file, "wb") as f:
      pickle.dump(results, f)
      print(f"  Saved results to {results_file}")
    return results


if __name__ == "__main__":
  start()
  # try_direct_export()
