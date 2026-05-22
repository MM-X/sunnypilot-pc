from opendbc.car import structs, get_safety_config
from opendbc.car.interfaces import CarInterfaceBase
from opendbc.car.changan.values import CAR, CarControllerParams, ChanganSafetyFlags, CanBus, MIN_ACC_SPEED
from opendbc.car.changan.carstate import CarState
from opendbc.car.changan.carcontroller import CarController


SteerControlType = structs.CarParams.SteerControlType
GearShifter = structs.CarState.GearShifter
NetworkLocation = structs.CarParams.NetworkLocation
SafetyModel = structs.CarParams.SafetyModel


class CarInterface(CarInterfaceBase):
  CarState = CarState
  CarController = CarController

  @staticmethod
  def _get_params(ret: structs.CarParams, candidate: CAR, fingerprint, car_fw, alpha_long: bool, is_release: bool, docs: bool) -> structs.CarParams:

    ret.brand = "changan"
    ret.safetyConfigs = [get_safety_config(SafetyModel.changan)]

    if candidate == CAR.CHANGAN_Z6:
      ret.safetyConfigs[0].safetyParam = int(ChanganSafetyFlags.CHANGAN_Z6_FLAG)
    elif candidate == CAR.CHANGAN_Z6_IDD:
      ret.safetyConfigs[0].safetyParam = int(ChanganSafetyFlags.CHANGAN_Z6_IDD_FLAG)

    ret.steerControlType = SteerControlType.angle

    ret.steerRatioRear = 0.0

    ret.steerActuatorDelay = 0.25
    ret.steerLimitTimer = 0.4

    ret.longitudinalTuning.kpBP = [0.0]
    ret.longitudinalTuning.kpV = [0.1]
    ret.longitudinalTuning.kiBP = [0.0]
    ret.longitudinalTuning.kiV = [0.0]

    ret.longitudinalActuatorDelay = 0.5
    ret.startAccel = 0.5
    ret.stopAccel = -2.0
    ret.stoppingDecelRate = 0.3
    ret.vEgoStarting = 0.5

    ret.openpilotLongitudinalControl = True
    ret.pcmCruise = False
    ret.minEnableSpeed = MIN_ACC_SPEED
    ret.minSteerSpeed = 0.0

    ret.radarUnavailable = True
    ret.networkLocation = NetworkLocation.gateway

    ret.autoResumeSng = True
    ret.enableBsm = False

    return ret

  def _update(self, c):
    ret = self.CS.update(self.cp, self.cp_cam)
    events = self.create_common_events(ret)
    ret.events = events.to_msg()
    return ret
