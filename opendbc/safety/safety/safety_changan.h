// Safety model for Changan Z6 / Z6 IDD
// Reversed from opendbc/car/changan/{carcontroller.py, changancan.py, values.py}
//
// Reverse engineering notes:
// - TX whitelist derived from changancan.py packer.make_can_msg() calls
// - RX whitelist derived from carstate.py cp.vl[] / cp_cam.vl[] lookups
// - Safety flags derived from values.py ChanganSafetyFlags enum
// - Bus topology: ESC=0 (pt), MRR=1, MPC/cam=2
//
// TX Messages (openpilot → car):
//   0x1BA (GW_1BA)  bus 0: EPS angle command (lateral)
//   0x17E (GW_17E)  bus 2: EPS torque sensor relay + lat active flag
//   0x244 (GW_244)  bus 0: ACC longitudinal command
//   0x307 (GW_307)  bus 0: ACC display / set speed relay
//   0x31A (GW_31A)  bus 0: IACC/HWA mode + AEB relay

#pragma once

#include "safety_declarations.h"

// ── Safety flags ──────────────────────────────────────────────────────────────
#define CHANGAN_Z6_FLAG     0x1U   // Standard Z6 (Veoneer MPC/radar solution)
#define CHANGAN_Z6_IDD_FLAG 0x4U   // Changan Z6 IDD variant

// ── Steering limits ───────────────────────────────────────────────────────────
// EPS_AngleCmd: factor 0.1 deg/LSB, max 980 deg → 9800 raw
// STEER_MAX from values.py = 980 deg
#define CHANGAN_STEER_ANGLE_MAX 9800    // 980 deg * 10
// Angle rate limit: 1.4 deg/frame (100Hz) = 14 raw/frame (from ANGLE_LIMITS in values.py)
#define CHANGAN_STEER_ANGLE_RATE 140    // 1.4 deg * 100 = 140 raw/s at 100Hz => 14/frame

// ── Acceleration limits ───────────────────────────────────────────────────────
// ACC_ACCTargetAcceleration: factor 0.05 m/s² per LSB
// ACCEL_MAX = 2.0 m/s²  → 40 raw
// ACCEL_MIN = -3.5 m/s² → -70 raw
#define CHANGAN_ACCEL_MAX   40    //  2.0 m/s² * 20
#define CHANGAN_ACCEL_MIN  -70    // -3.5 m/s² * 20

// ── RX messages ───────────────────────────────────────────────────────────────
// bus 0 (ESC/pt CAN)
#define MSG_GW_50    0x050U   // SRS seatbelt
#define MSG_GW_170   0x170U   // EPS actual torque
#define MSG_GW_17A   0x17AU   // ESP speed (Z6 IDD variant)
#define MSG_GW_17E   0x17EU   // EPS sensor (also TX relay on bus 2)
#define MSG_GW_180   0x180U   // SAS steering angle + rate
#define MSG_GW_187   0x187U   // ESP speed        (default Z6)
#define MSG_GW_196   0x196U   // EMS brake+accel  (default Z6)
#define MSG_GW_1A6   0x1A6U   // EMS brake (Z6 IDD)
#define MSG_GW_1C6   0x1C6U   // EMS accel (Z6 IDD)
#define MSG_GW_24F   0x24FU   // EPS fault status
#define MSG_GW_28B   0x28BU   // BCM doors + turn signals
#define MSG_GW_28C   0x28CU   // MFS steering wheel buttons
#define MSG_GW_338   0x338U   // TCU gear          (default Z6 / IDD)

// bus 2 (MPC/cam CAN)
#define MSG_GW_1BA   0x1BAU   // ACC angle cmd (also TX on bus 0 via relay)
#define MSG_GW_244   0x244U   // ACC longitudinal  (also TX on bus 0 via relay)
#define MSG_GW_307   0x307U   // ACC display       (also TX relay)
#define MSG_GW_31A   0x31AU   // IACC/HWA          (also TX relay)

// ── TX messages ───────────────────────────────────────────────────────────────
#define MSG_GW_1BA_TX  0x1BAU  // bus 0: lateral angle command
#define MSG_GW_17E_TX  0x17EU  // bus 2: EPS torque relay
#define MSG_GW_244_TX  0x244U  // bus 0: longitudinal command
#define MSG_GW_307_TX  0x307U  // bus 0: set speed relay
#define MSG_GW_31A_TX  0x31AU  // bus 0: IACC mode relay

// ── State variables ───────────────────────────────────────────────────────────
static uint32_t changan_safety_flags = 0U;
static int changan_desired_angle_last = 0;

// ── TX messages ───────────────────────────────────────────────────────────────
static const CanMsg CHANGAN_TX_MSGS[] = {
  {MSG_GW_1BA_TX, 0, 32},  // EPS lateral angle command
  {MSG_GW_17E_TX, 2, 8},   // EPS relay (bus 2 = cam)
  {MSG_GW_244_TX, 0, 32},  // ACC longitudinal
  {MSG_GW_307_TX, 0, 64},  // ACC display relay
  {MSG_GW_31A_TX, 0, 64},  // IACC mode relay
};

// ── RX allowed messages ───────────────────────────────────────────────────────
static RxCheck changan_rx_checks[] = {
  // bus 0 (pt)
  {.msg = {{MSG_GW_50,   0, 8,  .ignore_checksum = true, .ignore_counter = true, .frequency = 10U}, {0}, {0}}},
  {.msg = {{MSG_GW_170,  0, 8,  .ignore_checksum = true, .ignore_counter = true, .frequency = 100U}, {0}, {0}}},
  {.msg = {{MSG_GW_17E,  0, 8,  .ignore_checksum = true, .ignore_counter = true, .frequency = 100U}, {0}, {0}}},
  {.msg = {{MSG_GW_180,  0, 8,  .ignore_checksum = true, .ignore_counter = true, .frequency = 100U}, {0}, {0}}},
  {.msg = {{MSG_GW_24F,  0, 8,  .ignore_checksum = true, .ignore_counter = true, .frequency = 20U}, {0}, {0}}},
  {.msg = {{MSG_GW_28B,  0, 8,  .ignore_checksum = true, .ignore_counter = true, .frequency = 10U}, {0}, {0}}},
  {.msg = {{MSG_GW_28C,  0, 8,  .ignore_checksum = true, .ignore_counter = true, .frequency = 20U}, {0}, {0}}},
  {.msg = {{MSG_GW_187,  0, 8,  .ignore_checksum = true, .ignore_counter = true, .frequency = 100U},
           {MSG_GW_17A,  0, 8,  .ignore_checksum = true, .ignore_counter = true, .frequency = 100U},
           {0}}},
  {.msg = {{MSG_GW_196,  0, 8,  .ignore_checksum = true, .ignore_counter = true, .frequency = 100U},
           {MSG_GW_1A6,  0, 8,  .ignore_checksum = true, .ignore_counter = true, .frequency = 100U},
           {MSG_GW_1C6,  0, 8,  .ignore_checksum = true, .ignore_counter = true, .frequency = 100U}}},
  {.msg = {{MSG_GW_338,  0, 8,  .ignore_checksum = true, .ignore_counter = true, .frequency = 20U}, {0}, {0}}},
  // These messages also appear on bus 0 (from camera/EPS)
  {.msg = {{MSG_GW_1BA,  0, 32, .ignore_checksum = true, .ignore_counter = true, .frequency = 100U}, {0}, {0}}},
  {.msg = {{MSG_GW_244,  0, 32, .ignore_checksum = true, .ignore_counter = true, .frequency = 50U}, {0}, {0}}},
  {.msg = {{MSG_GW_307,  0, 64, .ignore_checksum = true, .ignore_counter = true, .frequency = 10U}, {0}, {0}}},
  {.msg = {{MSG_GW_31A,  0, 64, .ignore_checksum = true, .ignore_counter = true, .frequency = 10U}, {0}, {0}}},
  // bus 2 (cam)
  {.msg = {{MSG_GW_1BA,  2, 32, .ignore_checksum = true, .ignore_counter = true, .frequency = 100U}, {0}, {0}}},
  {.msg = {{MSG_GW_17E,  2, 8, .ignore_checksum = true, .ignore_counter = true, .frequency = 100U}, {0}, {0}}},
  {.msg = {{MSG_GW_244,  2, 32, .ignore_checksum = true, .ignore_counter = true, .frequency = 50U}, {0}, {0}}},
  {.msg = {{MSG_GW_307,  2, 64, .ignore_checksum = true, .ignore_counter = true, .frequency = 10U}, {0}, {0}}},
  {.msg = {{MSG_GW_31A,  2, 64, .ignore_checksum = true, .ignore_counter = true, .frequency = 10U}, {0}, {0}}},
};

// ── Safety hook: init ─────────────────────────────────────────────────────────
static safety_config changan_init(uint16_t param) {
  changan_safety_flags = param;
  changan_desired_angle_last = 0;
  return BUILD_SAFETY_CFG(changan_rx_checks, CHANGAN_TX_MSGS);
}

// ── Safety hook: TX allowed ───────────────────────────────────────────────────
// Validates each outgoing message against:
//   1. Steering angle limits  (GW_1BA)
//   2. Acceleration limits    (GW_244)
// Note: TX whitelist is already checked by safety_tx_hook in safety.h
static bool changan_tx_hook(const CANPacket_t *to_send) {
  const int addr = GET_ADDR(to_send);
  const int bus  = GET_BUS(to_send);
  bool tx = true;

  // ── GW_1BA: lateral angle command ────────────────────────────────────────
  // EPS_AngleCmd: bits [1..14], factor 0.1 deg/LSB, signed
  if ((addr == MSG_GW_1BA_TX) && (bus == 0)) {
    const int raw_angle = (int)(((GET_BYTE(to_send, 0) >> 1) | (GET_BYTE(to_send, 1) << 7)) & 0x3FFFU);
    // sign-extend 14-bit
    const int angle_cmd = (raw_angle & 0x2000) ? (raw_angle - 0x4000) : raw_angle;
    const uint8_t lat_active = GET_BYTE(to_send, 0) & 0x1U;

    if (lat_active != 0U) {
      // Angle delta limit (per-frame, ~14 raw = 1.4 deg at 100Hz)
      const int angle_delta = angle_cmd - changan_desired_angle_last;
      if ((angle_delta > CHANGAN_STEER_ANGLE_RATE) || (angle_delta < -CHANGAN_STEER_ANGLE_RATE)) {
        tx = false;
      }
      // Absolute angle limit
      if ((angle_cmd > CHANGAN_STEER_ANGLE_MAX) || (angle_cmd < -CHANGAN_STEER_ANGLE_MAX)) {
        tx = false;
      }
    }
    if (tx) {
      changan_desired_angle_last = angle_cmd;
    }
  }

  // ── GW_244: longitudinal acceleration command ─────────────────────────────
  // ACC_ACCTargetAcceleration: bits [4..15], factor 0.05 m/s², signed 12-bit
  if ((addr == MSG_GW_244_TX) && (bus == 0)) {
    const int raw_accel_u = (int)(((GET_BYTE(to_send, 0) >> 4) | (GET_BYTE(to_send, 1) << 4)) & 0xFFFU);
    const int accel_cmd = (raw_accel_u & 0x800) ? (raw_accel_u - 0x1000) : raw_accel_u;
    // raw unit = 0.05 m/s² per LSB; CHANGAN_ACCEL_MAX/MIN already in these units
    if ((accel_cmd > CHANGAN_ACCEL_MAX) || (accel_cmd < CHANGAN_ACCEL_MIN)) {
      tx = false;
    }
  }

  return tx;
}

// ── Safety hook: RX update ────────────────────────────────────────────────────
static void changan_rx_hook(const CANPacket_t *to_push) {
  (void)to_push;
}

// ── Safety hook: fwd ──────────────────────────────────────────────────────────
// Forward logic:
//   bus 0 (ESC/pt) → forward to bus 2 (MPC/cam): never (openpilot injects)
//   bus 2 (cam) → forward to bus 0 (ESC/pt): all except TX msgs intercepted by openpilot
//   bus 1 (MRR) → not forwarded
static int changan_fwd_hook(int bus_num, int addr) {
  int bus_fwd = -1;  // default: no forward

  if (bus_num == 2) {
    // Cam → pt: forward everything except what openpilot replaces
    const bool blocked = ((addr == MSG_GW_1BA_TX) ||
                          (addr == MSG_GW_244_TX) ||
                          (addr == MSG_GW_307_TX) ||
                          (addr == MSG_GW_31A_TX));
    if (!blocked) {
      bus_fwd = 0;
    }
  } else if (bus_num == 0) {
    // pt → cam: forward GW_17E (EPS relay) to bus 2
    if (addr == MSG_GW_17E) {
      bus_fwd = 2;
    }
  }

  return bus_fwd;
}

// ── Safety model registration ─────────────────────────────────────────────────
const safety_hooks changan_hooks = {
  .init          = changan_init,
  .rx            = changan_rx_hook,
  .tx            = changan_tx_hook,
  .fwd           = changan_fwd_hook,
};
