import copy
from opendbc.car.common.conversions import Conversions as CV
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.realtime import DT_CTRL
from opendbc.can import CANDefine, CANParser
from opendbc.car.interfaces import CarStateBase
from opendbc.car.changan.values import CAR, DBC, STEER_THRESHOLD, EPS_SCALE
from opendbc.car import Bus, structs
from openpilot.common.params import Params


SteerControlType = structs.CarParams.SteerControlType

TEMP_STEER_FAULTS = (0, 9, 11, 21, 25)
PERM_STEER_FAULTS = (3, 17)


class CarState(CarStateBase):
  def __init__(self, CP):
    super().__init__(CP)
    can_define = CANDefine(DBC[CP.carFingerprint][Bus.pt])

    self.shifter_values = can_define.dv["GW_338"]["TCU_GearForDisplay"]
    self.eps_torque_scale = EPS_SCALE[CP.carFingerprint] / 100.0
    self.cluster_speed_hyst_gap = CV.KPH_TO_MS / 2.0
    self.cluster_min_speed = CV.KPH_TO_MS / 2.0

    self.angle_offset = FirstOrderFilter(None, 60.0, DT_CTRL, initialized=False)

    self.prev_distance_button = 0
    self.distance_button = 0
    self.pcm_follow_distance = 0
    self.low_speed_lockout = False
    self.acc_type = 1
    self.cruiseEnable = False
    self.cruiseSpeed = 0
    self.buttonPlus = 0
    self.buttonReduce = 0

    self.counter_244 = 0
    self.counter_1ba = 0
    self.counter_17e = 0
    self.counter_307 = 0
    self.counter_31a = 0

    self.sigs244 = {}
    self.sigs1ba = {}
    self.sigs17e = {}
    self.sigs307 = {}
    self.sigs31a = {}

    self.steeringPressed = False
    self.steeringPressedMax = 0
    self.steeringPressedMin = 0

    self.iacc_enable_switch_button_pressed = 0
    self.iacc_enable_switch_button_prev = 0
    self.iacc_enable_switch_button_rising_edge = False

    try:
      self.cruise_control_mode = int(Params().get("CruiseControlMode", 0))
    except Exception:
      self.cruise_control_mode = 0

  def update(self, can_parsers) -> structs.CarState:
    cp = can_parsers[Bus.pt]
    cp_cam = can_parsers[Bus.cam]
    ret = structs.CarState()

    ret.doorOpen = any([cp.vl["GW_28B"]["BCM_DriverDoorStatus"]])
    ret.seatbeltUnlatched = cp.vl["GW_50"]["SRS_DriverBuckleSwitchStatus"] == 1
    ret.parkingBrake = False

    if self.CP.carFingerprint == CAR.CHANGAN_Z6_IDD:
      carspd = cp.vl["GW_17A"]["ESP_VehicleSpeed"]
    else:
      carspd = cp.vl["GW_187"]["ESP_VehicleSpeed"]

    speed = carspd if carspd <= 5 else ((carspd / 0.98) + 2)
    ret.vEgoRaw = speed * CV.KPH_TO_MS
    ret.vEgo, ret.aEgo = self.update_speed_kf(ret.vEgoRaw)
    ret.vEgoCluster = ret.vEgo

    ret.standstill = abs(ret.vEgoRaw) < 1e-3

    ret.steeringAngleOffsetDeg = 0
    ret.steeringAngleDeg = cp.vl["GW_180"]["SAS_SteeringAngle"]
    ret.steeringRateDeg = cp.vl["GW_180"]["SAS_SteeringAngleSpeed"]

    if self.CP.carFingerprint == CAR.CHANGAN_Z6_IDD:
      can_gear = int(cp.vl["GW_338"]["TCU_GearForDisplay"])
      ret.brakePressed = cp.vl["GW_1A6"]["EMS_BrakePedalStatus"] != 0
      ret.gasPressed = cp.vl["GW_1C6"]["EMS_RealAccPedal"] != 0
      self.steeringPressedMin = 1
      self.steeringPressedMax = 3
      ret.leftBlindspot = False
      ret.rightBlindspot = False
    else:
      can_gear = int(cp.vl["GW_338"]["TCU_GearForDisplay"])
      ret.brakePressed = cp.vl["GW_196"]["EMS_BrakePedalStatus"] != 0
      ret.gasPressed = cp.vl["GW_196"]["EMS_RealAccPedal"] != 0
      self.steeringPressedMin = 1
      self.steeringPressedMax = 6
      ret.leftBlindspot = False
      ret.rightBlindspot = False

    ret.gearShifter = self.parse_gear_shifter(self.shifter_values.get(can_gear, None))

    ret.leftBlinker, ret.rightBlinker = self.update_blinker_from_stalk(
      200, cp.vl["GW_28B"]["BCM_TurnIndicatorLeft"] == 1, cp.vl["GW_28B"]["BCM_TurnIndicatorRight"] == 1
    )

    ret.steeringTorque = cp.vl["GW_17E"]["EPS_MeasuredTorsionBarTorque"]
    ret.steeringTorqueEps = cp.vl["GW_170"]["EPS_ActualTorsionBarTorq"]

    if self.steeringPressed:
      if abs(ret.steeringTorque) < self.steeringPressedMin and abs(ret.steeringAngleDeg) < 90:
        self.steeringPressed = False
    else:
      if abs(ret.steeringTorque) > self.steeringPressedMax:
        self.steeringPressed = True
    ret.steeringPressed = self.steeringPressed

    ret.steerFaultTemporary = cp.vl["GW_24F"]["EPS_EPSFailed"] != 0 or cp.vl["GW_17E"]["EPS_LatCtrlAvailabilityStatus"] == 2

    self.iacc_enable_switch_button_pressed = cp.vl["GW_28C"]["GW_MFS_IACCenable_switch_signal"]
    self.iacc_enable_switch_button_rising_edge = self.iacc_enable_switch_button_pressed == 1 and self.iacc_enable_switch_button_prev == 0

    if self.cruiseEnable and (self.iacc_enable_switch_button_rising_edge or ret.brakePressed):
      self.cruiseEnable = False
    elif not self.cruiseEnable and self.iacc_enable_switch_button_rising_edge:
      self.cruiseEnable = True

    self.iacc_enable_switch_button_prev = self.iacc_enable_switch_button_pressed

    if self.cruiseEnable and not self.cruiseEnablePrev:
      self.cruiseSpeed = speed if self.cruiseSpeed == 0 or self.cruise_control_mode == 0 else self.cruiseSpeed

    if cp.vl["GW_28C"]["GW_MFS_RESPlus_switch_signal"] == 1 and self.buttonPlus == 0 and self.cruiseEnable:
      self.cruiseSpeed = ((self.cruiseSpeed // 5) + 1) * 5

    if cp.vl["GW_28C"]["GW_MFS_SETReduce_switch_signal"] == 1 and self.buttonReduce == 0 and self.cruiseEnable:
      self.cruiseSpeed = max((((self.cruiseSpeed // 5) - 1) * 5), 0)

    self.cruiseEnablePrev = self.cruiseEnable
    self.buttonPlus = cp.vl["GW_28C"]["GW_MFS_RESPlus_switch_signal"]
    self.buttonReduce = cp.vl["GW_28C"]["GW_MFS_SETReduce_switch_signal"]

    ret.accFaulted = cp_cam.vl["GW_244"]["ACC_ACCMode"] == 7 or cp_cam.vl["GW_31A"]["ACC_IACCHWAMode"] == 7
    ret.cruiseState.available = cp_cam.vl["GW_31A"]["ACC_IACCHWAEnable"] == 1
    ret.cruiseState.speed = self.cruiseSpeed * CV.KPH_TO_MS
    cluster_set_speed = self.cruiseSpeed

    if ret.cruiseState.speed != 0:
      ret.cruiseState.speedCluster = cluster_set_speed * CV.KPH_TO_MS

    ret.stockFcw = cp_cam.vl["GW_244"]["ACC_FCWPreWarning"] == 1
    ret.cruiseState.standstill = ret.standstill
    ret.cruiseState.enabled = self.cruiseEnable
    ret.genericToggle = False
    ret.stockAeb = cp_cam.vl["GW_244"]["ACC_AEBCtrlType"] > 0

    self.sigs244 = copy.copy(cp_cam.vl["GW_244"])
    self.sigs1ba = copy.copy(cp_cam.vl["GW_1BA"])
    self.sigs17e = copy.copy(cp.vl["GW_17E"])
    self.sigs307 = copy.copy(cp_cam.vl["GW_307"])
    self.sigs31a = copy.copy(cp_cam.vl["GW_31A"])
    self.counter_244 = cp_cam.vl["GW_244"]["ACC_RollingCounter_24E"]
    self.counter_1ba = cp_cam.vl["GW_1BA"]["ACC_RollingCounter_1BA"]
    self.counter_17e = cp.vl["GW_17E"]["EPS_RollingCounter_17E"]
    self.counter_307 = cp_cam.vl["GW_307"]["ACC_RollingCounter_35E"]
    self.counter_31a = cp_cam.vl["GW_31A"]["ACC_RollingCounter_36D"]

    self.prev_distance_button = self.distance_button
    self.distance_button = cp_cam.vl["GW_307"]["ACC_DistanceLevel"]

    return ret

  @staticmethod
  def get_can_parsers(CP):
    pt_messages = [
      ("GW_28B", 0),
      ("GW_50", 0),
      ("GW_187", 0),
      ("GW_17A", 0),
      ("GW_17E", 0),
      ("GW_170", 0),
      ("GW_180", 0),
      ("GW_338", 0),
      ("GW_1BA", 0),
      ("GW_244", 0),
      ("GW_307", 0),
      ("GW_31A", 0),
    ]
    cam_messages = [
      ("GW_244", 0),
      ("GW_1BA", 0),
      ("GW_307", 0),
      ("GW_31A", 0),
    ]
    return {
      Bus.pt: CANParser(DBC[CP.carFingerprint][Bus.pt], pt_messages, 0),
      Bus.cam: CANParser(DBC[CP.carFingerprint][Bus.pt], cam_messages, 2),
    }
