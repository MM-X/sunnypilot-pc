from opendbc.car.changan.can_crc import CanCrc

can_crc = CanCrc()


def create_244_command_a05(packer, accel, counter, longActive, accTrq):
  values = {
    "ACC_ACCTargetAcceleration": accel,
    "ACC_AEBTargetDeceleration": -0.0005,
    "unknown": 1,
    "ACC_CDDActive": 1 if accel < 0 else 0,
    "ACC_LDWStatus": 0,
    "ACC_LDWVibrationWarningReq": 0,
    "ACC_LKAStatus": 0,
    "ACC_TextInfoForDriver": 0,
    "ACC_RollingCounter_24E": counter,
    "ACC_CRCCheck_24E": 0,
    "ACC_RollingCounter_25E": counter,
    "ACC_CRCCheck_25E": 0,
    "ADS_RollingCounter_244": counter,
    "ADS_CRCCheck_244": 0,
    "ACC_ACCMode": 3 if longActive else 2,
    "ACC_AEBActive": 0,
    "ACC_AccTrqReq": (120 + (accel - 0.05) / 0.05 * 30) if accel > 0 else 5005,
    "ACC_AccTrqReqActive": 1 if longActive and accel > 0 else 0,
    "ACC_FCWPreWarning": 0,
    "ACC_FCWLatentWarning": 0,
    "ACC_AWBActive": 0,
    "ACC_AEBCtrlType": 0,
    "ACC_LatTakeoverReq": 0,
    "ACC_LngTakeOverReq": 0,
    "ADS_EOMTOReq": 0,
    "ACC_HandsOnReq": 0,
  }
  dat = packer.make_can_msg("GW_244", 0, values)[1]
  values["ACC_CRCCheck_24E"] = can_crc.crc_calculate_crc8(dat[:7])
  values["ACC_CRCCheck_25E"] = can_crc.crc_calculate_crc8(dat[8:15])
  dat = packer.make_can_msg("GW_244", 0, values)[1]
  values["ADS_CRCCheck_244"] = can_crc.crc16_ccitt_false(dat[:54])
  return packer.make_can_msg("GW_244", 0, values)


def create_244_command(packer, msg: dict, accel, counter, longActive, accTrq, vEgoRaw):
  values = msg.copy()
  values.update(
    {
      "ACC_ACCTargetAcceleration": accel,
      "ACC_CDDActive": 1 if longActive and accel < 0 else 0,
      "ACC_RollingCounter_24E": counter,
      "ACC_RollingCounter_25E": counter,
      "ACC_ACCMode": 3 if longActive else 2,
      "ACC_AccTrqReq": accTrq,
      "ACC_AccTrqReqActive": 1 if longActive and accel >= 0 else 0,
    }
  )
  dat = packer.make_can_msg("GW_244", 0, values)[1]
  values["ACC_CRCCheck_24E"] = can_crc.crc_calculate_crc8(dat[:7])
  values["ACC_CRCCheck_25E"] = can_crc.crc_calculate_crc8(dat[8:15])
  return packer.make_can_msg("GW_244", 0, values)


def create_244_command_idd(packer, msg: dict, accel, counter, longActive, accTrq, vEgoRaw):
  values = msg.copy()
  values.update(
    {
      "ACC_ACCTargetAcceleration": accel,
      "ACC_CDDActive": 1 if longActive and accel < 0 else 0,
      "ACC_RollingCounter_24E": counter,
      "ACC_RollingCounter_25E": counter,
      "ACC_ACCMode": 3 if longActive else 2,
      "ACC_AccTrqReq": accTrq,
      "ACC_AccTrqReqActive": 1 if longActive and accel >= 0 else 0,
      "ACC_DecToStop": 1 if longActive and accel < 0 and vEgoRaw == 0 else 0,
      "ACC_AWBActive": 0,
      "ACC_AEBCtrlType": 0,
      "ACC_TextInfoForDriver": 0,
      "ACC_Driveoff_Request": 0,
      "ACC_FCWPreWarning": 0,
      "ACC_FCWLatentWarning": 0,
      "ACC_LatTakeoverReq": 0,
      "ACC_LngTakeOverReq": 0,
      "ACC_HandsOnReq": 0,
    }
  )
  dat = packer.make_can_msg("GW_244", 0, values)[1]
  values["ACC_CRCCheck_24E"] = can_crc.crc_calculate_crc8(dat[:7])
  values["ACC_CRCCheck_25E"] = can_crc.crc_calculate_crc8(dat[8:15])
  return packer.make_can_msg("GW_244", 0, values)


def create_1BA_command(packer, msg: dict, angle, latCtrlActive, counter):
  values = {
    s: msg.get(s, 0)
    for s in [
      "ACC_CRCCheck_1BA",
      "ACC_RollingCounter_1BA",
      "EPS_LatCtrlActive",
      "EPS_AngleCmd",
      "ACC_MotorTorqueMinLimitRequest",
      "ACC_MotorTorqueMaxLimitRequest",
    ]
  }
  values.update(
    {
      "EPS_AngleCmd": angle,
      "EPS_LatCtrlActive": latCtrlActive,
      "ACC_RollingCounter_1BA": counter,
    }
  )
  dat = packer.make_can_msg("GW_1BA", 0, values)[1]

  values["ACC_CRCCheck_1BA"] = can_crc.crc_calculate_crc8(dat[:7])
  return packer.make_can_msg("GW_1BA", 0, values)


def create_17E_command(packer, msg: dict, longActive, counter):
  values = {
    s: msg.get(s, 0)
    for s in [
      "EPS_CRCCheck_17E",
      "EPS_RollingCounter_17E",
      "EPS_LatCtrlAvailabilityStatus",
      "EPS_LatCtrlActive",
      "EPS_Handwheel_Relang_Valid",
      "EPS_MeasuredTorsionBarTorqValid",
      "EPS_Handwheel_Relang",
      "EPS_Pinionang",
      "EPS_Pinionang_Valid",
      "EPS_ADS_Abortfeedback",
      "EPS_MeasuredTorsionBarTorque",
    ]
  }
  values.update(
    {
      "EPS_MeasuredTorsionBarTorque": msg.get("EPS_MeasuredTorsionBarTorque", 0) + 1 if longActive else msg.get("EPS_MeasuredTorsionBarTorque", 0),
      "EPS_RollingCounter_17E": counter,
    }
  )
  dat = packer.make_can_msg("GW_17E", 0, values)[1]
  values["EPS_CRCCheck_17E"] = can_crc.crc_calculate_crc8(dat[:7])

  return packer.make_can_msg("GW_17E", 2, values)


def create_307_command(packer, msg: dict, counter, cruiseSpeed):
  values = {
    s: msg.get(s, 0)
    for s in [
      "ACC_RLaneDistanceFus",
      "ACC_LLaneDistanceFus",
      "ACC_RRLaneDis",
      "ACC_LLLaneDis",
      "ACC_Target7ZoneID",
      "ACC_Target7HeadingAngle",
      "ACC_Target7LatRange",
      "ACC_Target7LngRange",
      "ACC_Target7Direction",
      "ACC_Target7Type",
      "ACC_Target7ID",
      "ACC_Target7Detection",
      "ACC_Target6ZoneID",
      "ACC_Target6HeadingAngle",
      "ACC_Target6LatRange",
      "ACC_Target6LngRange",
      "ACC_Target6Direction",
      "ACC_Target6Type",
      "ACC_Target6ID",
      "ACC_Target6Detection",
      "ACC_CRCCheck_35F",
      "ACC_RollingCounter_35F",
      "ACC_CSLAEnableStatus",
      "ACC_IACCProhibitionTime",
      "ACC_CSLSetReq",
      "ACC_VehicleStartRemindSts",
      "ACC_CRCCheck_344",
      "ACC_RollingCounter_344",
      "ACC_CRCCheck_322",
      "ACC_RollingCounter_322",
      "ACC_ACCTargetRelSpd",
      "ACC_FRadarCalibrationStatus",
      "ACC_CRCCheck_35E",
      "ACC_RollingCounter_35E",
      "ACC_AEBEnable",
      "ACC_FCWSettingStatus",
      "ACC_TimeGapSet",
      "ACC_DistanceLevel",
      "ACC_ObjValid",
      "ACC_SetSpeed",
    ]
  }
  values.update(
    {
      "ACC_SetSpeed": cruiseSpeed,
      "ACC_RollingCounter_35E": counter,
      "ACC_RollingCounter_322": counter,
      "ACC_RollingCounter_344": counter,
      "ACC_RollingCounter_35F": counter,
    }
  )
  dat = packer.make_can_msg("GW_307", 0, values)[1]
  values["ACC_CRCCheck_35E"] = can_crc.crc_calculate_crc8(dat[:7])
  values["ACC_CRCCheck_322"] = can_crc.crc_calculate_crc8(dat[8:15])
  values["ACC_CRCCheck_344"] = can_crc.crc_calculate_crc8(dat[16:23])
  values["ACC_CRCCheck_35F"] = can_crc.crc_calculate_crc8(dat[24:31])
  return packer.make_can_msg("GW_307", 0, values)


def create_31A_command(packer, msg: dict, counter, longActive, steeringPressed):
  values = {
    s: msg.get(s, 0)
    for s in [
      "ACC_CRCCheck_367",
      "ACC_RollingCounter_367",
      "ACC_LatPathHeadingAngle",
      "ACC_LatPathDY",
      "ACC_ELKEnableStatus",
      "ACC_ELKInterventionMode",
      "ACC_ELKMode",
      "ACC_CRCCheck_30D",
      "ACC_RollingCounter_30D",
      "ACC_HighBeamControl",
      "ACC_RRLaneDetection",
      "ACC_LLLaneDetection",
      "ACC_TargetBasedLateralControl",
      "ACC_DriverHandsOffStatus",
      "ACC_IACCHWATextInfoForDriver",
      "ACC_IACCHWAMode",
      "ACC_CRCCheck_30A",
      "ACC_RollingCounter_30A",
      "ACC_LaneChangeStatus",
      "ACC_RoadCurvatureFar",
      "ACC_RoadCurvatureNear",
      "ACC_RoadCurvature",
      "ACC_LLaneMarkerType",
      "ACC_HostLaneLeftStatus",
      "ACC_HostLaneRightStatus",
      "ACC_IACCHWAEnable",
      "ACC_RLaneMarkerType",
      "ACC_CRCCheck_36D",
      "ACC_RollingCounter_36D",
      "ACC_FRadarFailureStatus",
      "ACC_Voiceinfo",
      "ACC_AEBTargetmode",
      "ACC_AEBTextInfo",
      "ACC_AEBStatus",
      "ACC_ELKAlert",
      "ACC_AEBTargetLatRange",
      "ACC_AEBTargetRelSpeed",
      "ACC_AEBTargetLngRange",
    ]
  }
  iacc_mode = 1
  if longActive:
    if steeringPressed:
      iacc_mode = 4
    else:
      iacc_mode = 3
  values.update(
    {
      "ACC_IACCHWAMode": iacc_mode,
      "ACC_TargetBasedLateralControl": 2 if longActive and not steeringPressed else 0,
      "ACC_AEBTextInfo": 0,
      "ACC_IACCHWATextInfoForDriver": 0,
      "ACC_ELKAlert": 0,
      "ACC_RollingCounter_36D": counter,
      "ACC_RollingCounter_30A": counter,
      "ACC_RollingCounter_30D": counter,
      "ACC_RollingCounter_367": counter,
    }
  )
  dat = packer.make_can_msg("GW_31A", 0, values)[1]
  values["ACC_CRCCheck_36D"] = can_crc.crc_calculate_crc8(dat[:7])
  values["ACC_CRCCheck_30A"] = can_crc.crc_calculate_crc8(dat[8:15])
  values["ACC_CRCCheck_30D"] = can_crc.crc_calculate_crc8(dat[16:23])
  values["ACC_CRCCheck_367"] = can_crc.crc_calculate_crc8(dat[24:31])
  return packer.make_can_msg("GW_31A", 0, values)
