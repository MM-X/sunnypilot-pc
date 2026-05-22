from collections import defaultdict
from dataclasses import dataclass, field
from enum import IntFlag

from opendbc.car import Bus, DbcDict, PlatformConfig, Platforms, CarSpecs, structs
from opendbc.car import AngleSteeringLimits
from opendbc.car.docs_definitions import CarDocs, CarParts, CarHarness
from opendbc.car.fw_query_definitions import FwQueryConfig, Request, StdQueries


Ecu = structs.CarParams.Ecu
MIN_ACC_SPEED = 19.0  # mph to m/s


class CanBus:
  ESC = 0
  MRR = 1
  MPC = 2


class CarControllerParams:
  ACCEL_MAX = 2.0  # m/s2
  ACCEL_MIN = -3.5  # m/s2

  STEER_STEP = 1
  STEER_MAX = 980
  STEER_ERROR_MAX = 1200

  ANGLE_LIMITS: AngleSteeringLimits = AngleSteeringLimits(
    980,
    ([10, 40], [1.4, 1.4]),
    ([10, 40], [1.4, 1.4]),
  )

  def __init__(self, CP):
    if CP.lateralTuning.which == "torque":
      self.STEER_DELTA_UP = 25
      self.STEER_DELTA_DOWN = 30
    else:
      self.STEER_DELTA_UP = 15
      self.STEER_DELTA_DOWN = 35


class ChanganSafetyFlags(IntFlag):
  CHANGAN_Z6_FLAG = 0x1
  CHANGAN_Z6_IDD_FLAG = 0x4
  IDD_VARIANT = 0x8


class ChanganFlags(IntFlag):
  IDD = 0x2


@dataclass
class ChanganCarDocs(CarDocs):
  package: str = "All"
  car_parts: CarParts = field(default_factory=CarParts.common([CarHarness.custom]))


@dataclass
class ChanganPlatformConfig(PlatformConfig):
  dbc_dict: DbcDict = field(default_factory=lambda: {Bus.pt: "changan_pt"})


class CAR(Platforms):
  CHANGAN_Z6 = ChanganPlatformConfig(
    [
      ChanganCarDocs("Changan Z6"),
    ],
    CarSpecs(mass=2205, wheelbase=2.80, steerRatio=15, tireStiffnessFactor=0.444),
  )

  CHANGAN_Z6_IDD = ChanganPlatformConfig(
    [
      ChanganCarDocs("Changan Z6 Idd"),
    ],
    CarSpecs(mass=2205, wheelbase=2.80, steerRatio=15, tireStiffnessFactor=0.444),
  )


FW_QUERY_CONFIG = FwQueryConfig(
  requests=[
    Request(
      [StdQueries.MANUFACTURER_SOFTWARE_VERSION_REQUEST],
      [StdQueries.MANUFACTURER_SOFTWARE_VERSION_RESPONSE],
      bus=CanBus.ESC,
    ),
  ],
)

STEER_THRESHOLD = 15

EPS_SCALE = defaultdict(lambda: 73)

DBC = CAR.create_dbc_map()

NO_STOP_TIMER_CAR = {CAR.CHANGAN_Z6, CAR.CHANGAN_Z6_IDD}
