from driver.esen_driver import eSENDriver
from driver.gpaw_driver import GPAWDFTDriver
from driver.mace_driver import MACEDriver
from driver.mattersim_driver import MatterSimDriver
from driver.nequix_driver import NequixDriver
from driver.orb_driver import Orbv3Driver, Orbv2Driver
from driver.painn_driver import PaiNNDriver
from driver.schnet_driver import SchNetDriver
from driver.sevennet_driver import SevenNetDriver
# from driver.uma_driver import UMADriver
from driver.nequip_driver import NequipDriver

drivers = dict({
  "orbv3": Orbv3Driver(name="orbv3"),
  "orbv2": Orbv2Driver(name="orbv2"),
  # "uma": UMADriver(name="uma"),
  "esen": eSENDriver(name="esen"),
  "neqip_allegro": NequipDriver(name="neqip_allegro"),
  "nequix": NequixDriver(name="nequix"),
  "mace": MACEDriver(name="mace"),
  "dft_gpaw": GPAWDFTDriver(name="dft_gpaw"),
  "mattersim": MatterSimDriver(name="mattersim"),
  "sevennet": SevenNetDriver(name="sevennet"),
  "painn": PaiNNDriver(name="painn"),
  "schnet": SchNetDriver(name="schnet"),
})