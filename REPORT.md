# CLAUDE.md — sunnypilot-pc 仓库指南

本文件由 Claude 编写，用于帮助后续迁移工作。仓库为 [MM-X/sunnypilot-pc](https://github.com/MM-X/sunnypilot-pc)，当前分支 `master-tici`（comma 三代设备 TICI 分支）。

## 1. 仓库布局

仓库根 `/home/mx/sunnypilot-pc` 采用**根级包 + `openpilot/` 符号链接包**的双布局：

- 真实代码位于根级目录：`selfdrive/`、`system/`、`common/`、`sunnypilot/`、`cereal/`、`third_party/`、`tools/`、`msgq/`、`opendbc/`、`panda/`、`rednose/`、`tinygrad/`。
- `openpilot/` 是一个 Python 包目录，内部用符号链接指回根级包：
  - `openpilot/common -> ../common`
  - `openpilot/selfdrive -> ../selfdrive/`
  - `openpilot/system -> ../system/`
  - `openpilot/sunnypilot -> ../sunnypilot`
  - `openpilot/third_party -> ../third_party`
  - `openpilot/tools -> ../tools`
- 因此 `import openpilot.selfdrive.modeld` 与 `selfdrive/modeld/` 是**同一份文件**。迁移根级 `selfdrive/modeld/` 即等于迁移 `openpilot/selfdrive/modeld/`。
- `tinygrad_repo/` 是独立 git 仓库（真实检出，非符号链接），为推理后端。
- `msgq_repo/`、`opendbc_repo/`、`rednose_repo/`、`teleoprtc_repo/` 等为各自上游子仓库的检出。

## 2. 构建系统

- 顶层 `SConstruct` 用 SCons 递归调用各包的 `SConscript`：
  - `common/SConscript`、`cereal/SConscript`、`panda/SConscript`、`rednose/SConscript`
  - `system/ubloxd`、`system/loggerd`、`system/logcatd`、`system/proclogd`、`system/camerad/SConscript`
  - `third_party/SConscript`
  - `selfdrive/SConscript`（进一步递归到 `selfdrive/modeld/SConscript`）
  - `sunnypilot/SConscript`
- modeld 的编译入口：`selfdrive/SConscript` → `selfdrive/modeld/SConscript`。
- `pyproject.toml` 声明运行时依赖，含 `onnx >= 1.14.0`；pytest 配置忽略 `openpilot/`、`tinygrad_repo/` 等子目录。

## 3. 当前 modeld 架构（旧：拆分模型 + C++ + OpenCL）

`selfdrive/modeld/` 现状：

- **C++ 推理内核**：`models/commonmodel.cc` + `models/commonmodel.h`，编译为 `libcommonmodel.a`。
- **Cython 绑定**：`models/commonmodel_pyx.pyx` / `.pxd`，编译为 `models/commonmodel_pyx.so`；运行时通过 `DrivingModelFrame`、`CLContext` 暴露给 Python。
- **OpenCL transforms**：`transforms/transform.{cc,h,cl}`、`transforms/loadyuv.{cc,h,cl}`，把 YUV/相机帧变换为模型输入。
- **runners**：`runners/tinygrad_helpers.py`（`qcom_tensor_from_opencl_address`，把 OpenCL buffer 桥接给 tinygrad）。
- **模型（拆分式）**：
  - `driving_policy.onnx` / `driving_policy_tinygrad.pkl` / `driving_policy_metadata.pkl`
  - `driving_vision.onnx` / `driving_vision_tinygrad.pkl` / `driving_vision_metadata.pkl`
  - `dmonitoring_model.onnx` / `_tinygrad.pkl` / `_metadata.pkl`
  - `supercombo_*`、`driving_vision_policy.pkl` 等历史产物。
- **SConscript 行为**（70 行）：编译 `commonmodel_pyx.so`；为 `driving_vision`、`driving_policy`、`dmonitoring_model` 三个模型用 `tinygrad_repo/examples/openpilot/compile3.py` 生成 `_tinygrad.pkl`；可选地在有 USB GPU 时编译 `big_driving_policy`、`big_driving_vision`。
- **运行时入口**：
  - `modeld.py`（驾驶模型）`PROCESS_NAME="selfdrive.modeld.modeld"`，加载 `driving_vision` + `driving_policy` 两个 pkl，继承 `sunnypilot.modeld.modeld_base.ModelStateBase`。
  - `dmonitoringmodeld.py`（驾驶员监控模型）。
- `sunnypilot/modeld/` 与 `sunnypilot/modeld_v2/` 均存在但为旧版（`modeld_v2` 缺 `modeld_base.py`）。

## 4. 目标架构（源仓库 `sunnypilot-pc-new`，更新版）

源仓库 `/home/mx/sunnypilot-pc-new/openpilot/selfdrive/modeld/` 为新版，差异巨大：

- **纯 tinygrad JIT，无 C++ / 无 OpenCL / 无 runners**：移除 `commonmodel*`、`transforms/`、`runners/`。
- **新增编译脚本**：
  - `compile_modeld.py`：从 `driving_supercombo.onnx` 生成融合的 `driving_tinygrad.pkl`（含 warp + policy JIT、metadata、output_slices），导出 `make_input_queues`、`WARP_INPUTS`、`POLICY_INPUTS` 供运行时调用。
  - `compile_dm_warp.py`：按相机分辨率生成 `dm_warp_{W}x{H}_tinygrad.pkl`。
  - `helpers.py`：`dump_oob`/`load_oob`（pickle 协议 5 带外缓冲，避免大 pkl 内存峰值）、`usbgpu_present()`、`modeld_pkl_path()`、`get_tg_input_devices()`、`TG_INPUT_DEVICES_PATH`。
- **模型（融合式）**：
  - `driving_supercombo.onnx`（约 61 MB）、`big_driving_supercombo.onnx`（占位 135 B）、`dmonitoring_model.onnx`。
- **大 pkl 分块**：经 `openpilot.common.file_chunker` 分块/重组（绕过 git 单文件 100 MB 限制）。
- **运行时**：`modeld.py` 继承 `openpilot.sunnypilot.modeld_v2.modeld_base.ModelStateBase`，用 `Tensor.from_blob` 直接引用 visionipc 缓冲，`load_oob(open_file_chunked(...))` 加载 JIT。
- `PROCESS_NAME="openpilot.selfdrive.modeld.modeld"`（当前为 `"selfdrive.modeld.modeld"`，需注意 messaging 权限/匹配）。
- SConscript（140 行）探测设备选 `CUDA`/`QCOM`/`CPU:LLVM`，生成 `driving_tinygrad.pkl`(分块) + `dmonitoring_model_metadata.pkl` + 各分辨率 `dm_warp_*_tinygrad.pkl` + `dmonitoring_model_tinygrad.pkl`，并写 `tg_input_devices.json`。

## 5. 迁移引入的跨包依赖（当前仓库缺失或过旧）

源 `modeld` 不再自包含，迁移会牵连以下**modeld 之外**的文件：

| 依赖路径 | 当前状态 | 说明 |
|---|---|---|
| `common/file_chunker.py` | **缺失** | 大 pkl 分块/重组，SConscript 与运行时都依赖 |
| `system/camerad/cameras/nv12_info.py` (+ `.h`) | **缺失** | NV12 帧布局信息，warp/compile 依赖 |
| `common/hardware/hw.py` (+ `base.py`、`pc/`、`tici/`、`usb.py`) | **缺失**（当前为 `system/hardware/`） | SConscript 把它列为编译依赖 |
| `sunnypilot/modeld_v2/`（`modeld_base.py`、`warp.py`、`parse_model_outputs.py` 等） | 旧版，缺关键文件 | `modeld.py` 运行时继承 `ModelStateBase` |
| `sunnypilot/models/helpers.py` (`plan_x_idxs_helper`) | 存在 | `fill_model_msg.py` 依赖，需核对接口一致 |
| `common/transformations/camera.py` (`_ar_ox_fisheye`/`_os_fisheye`) | 存在 | 已具备所需相机配置 |
| `common/transformations/model.py` (`DM_INPUT_SIZE`/`MEDMODEL_INPUT_SIZE`/`dmonitoringmodel_intrinsics`) | 存在 | 已具备 |
| `tinygrad_repo` 版本 | 真实 git @ d2bb1bcb9 | 需支持 `Tensor.from_blob`、`engine.jit.TinyJit`、`Device.get_available_devices`、`examples/openpilot/compile3.py` |
| `system/camerad/SConscript` | 存在 | 若 `nv12_info` 需纳入 camerad 编译则需同步 |

## 6. 迁移风险与注意点

1. **非孤立迁移**：只替换 `selfdrive/modeld/` 会导致运行时 `ImportError`（`file_chunker`、`nv12_info`、`modeld_v2.modeld_base` 缺失）。必须连带迁移依赖闭包。
2. **C++/OpenCL 清理**：新版移除 `commonmodel*`、`transforms/`、`runners/`；需确认无其他包引用这些产物（如 `sunnypilot/modeld/`、UI、tests）。
3. **`PROCESS_NAME` 变更**：`selfdrive.modeld.modeld` → `openpilot.selfdrive.modeld.modeld`，影响 messaging 注册/权限配置。
4. **模型文件大**：`driving_supercombo.onnx` 61 MB，生成的 pkl 可能 >100 MB，需 `file_chunker` 分块入 git。
5. **tinygrad 版本**：源的 `tinygrad_repo` 位于 `sunnypilot-pc-new/tinygrad_repo`（非独立 git 仓库，仅目录副本），版本未确认；需与当前 git 版本比对，必要时升级。
6. **PC 平台适配**：本仓库面向 PC（`master-tici`），源在 PC 上走 `DEV=CPU:LLVM`；USB GPU/`big_` 分支在 PC 上通常不构建，需裁剪。
7. **测试**：`selfdrive/modeld/tests/test_modeld.py` 存在；迁移后需更新或新建自检。

## 7. 常用命令（建议）

- 构建：`scons -j8`（根目录）
- 仅 modeld：进入对应 SConscript 路径后 `scons selfdrive/modeld`
- 运行时环境：`source launch_env.sh`
- 启动：`./launch_openpilot.sh`


commit 2c334ede443d7391d27575af8c854a095ba702a8 (grafted, HEAD -> master, origin/master, origin/HEAD)
Author: Jason Wen <haibin.wen3@gmail.com>
Date:   Mon Jul 20 15:05:19 2026 -0400

    mapd: ignore in plannerd health check (#1880)

    * mapd: ignore in plannerd health check

    * more


modeld 消息消费者迁移差异报告
源仓库：/home/mx/sunnypilot-pc-new/openpilot/...（新架构）
当前仓库：/home/mx/sunnypilot-pc/...（master-tici，modeld 已迁移）

全局观察：源仓库把 cereal 拆成 openpilot.cereal + opendbc.car.structs，所有消费者文件都有这一行 import 改动；这不属于 modeld 消息消费差异，但会随每个文件一起迁移。下面只报告与 modeld 输出（modelV2 / drivingModelData / cameraOdometry / modelDataV2SP）直接相关的差异。

一、必须迁移（与 modeld 新架构直接相关）
1. selfdrive/controls/radard.py — 核心差异，必须迁移
源：openpilot/selfdrive/controls/radard.py vs 当前：selfdrive/controls/radard.py
性质：消费 modelV2.leadsV3 字段，且 schema 用法变化
关键改动：
lead_msg.prob → 改用过滤后的 lead_prob（新增 lead_prob_filters 一阶滤波器，两个 lead 各一个），从 sm['modelV2'].leadsV3[i].prob 取
Track.get_RadarState(model_prob) 签名变化；is_potential_fcw 删除；measured 字段移除
leadOne.status / leadTwo.status → present（字段重命名，modeld 发布的 RadarState 也变了，下游消费者必须同步）
get_RadarState_from_vision(...) 新增 lead_prob 参数
self.radar_state.carStateMonoTime 字段被删除（modeld 不再发该 mono time 关联）
与 modeld 关联：直接。lead 概率滤波是对 modelV2 leadsV3 的新处理；status→present 是 schema 变更级联。
迁移必要性：必须。否则 lead 检测逻辑与 modeld 发布的 RadarState/leadData schema 不匹配，runtime 报错或漏检。
2. selfdrive/controls/lib/longitudinal_planner.py — 核心差异，必须迁移
源 vs 当前：同路径
性质：消费 modelV2 的 position/velocity/acceleration/disengagePredictions/action
关键改动：
删除 parse_model(model_msg) 静态方法（不再从 modelV2.position.x/velocity.x/acceleration.x interp 到 MPC 时间轴）
MPC update(...) 签名变化：不再传入 x, v, a, j，只传 radarState, v_cruise, personality —— 说明 modeld 输出路径不再被 longitudinal MPC 直接消费，MPC 内部重构
throttle_prob 仍从 sm['modelV2'].meta.disengagePredictions.gasPressProbs 取（消费未变）
output_a_target_e2e = sm['modelV2'].action.desiredAcceleration（消费未变）
longitudinalPlan.hasLead = sm['radarState'].leadOne.status → .present（随 radard.py 的 schema 变更）
新增 LongitudinalPlanSource 枚举、is_e2e(sm) 替代 mode == 'blended' 分支
mpc.mode = 'acc'/'blended' 整套删除 —— modeld 不再驱动 mode 切换
构造签名 __init__(self, CP, CP_SP, ...) 新增 CP_SP
与 modeld 关联：直接。modelV2 路径消费从 longitudinal_planner 上移/下沉到别处；status→present 联动。
迁移必要性：必须。
3. selfdrive/controls/controlsd.py — 核心差异，必须迁移
源 vs 当前：同路径
性质：订阅 modelV2，新增订阅 lateralManeuverPlan（modeld 链路下游新消息）
关键改动：
self.sm 新增订阅 'lateralManeuverPlan'、'liveDelay'（原来 liveDelay 已在但顺序/位置变化）
model_v2.action.desiredCurvature 消费变化：当 sm.valid['lateralManeuverPlan'] 时改用 sm['lateralManeuverPlan'].desiredCurvature，否则回退到 model_v2.action.desiredCurvature —— 新 modeld 架构把 desiredCurvature 从 modelV2 迁到 lateralManeuverPlan
lat_delay = self.sm["liveDelay"].lateralDelay + LAT_SMOOTH_SECONDS（LAT_SMOOTH_SECONDS 从 openpilot.selfdrive.modeld.modeld 导入，modeld 新常量）
LaC.update(...) 签名新增 lat_delay 参数；返回 steer, lateral_output, lac_log（原 steeringAngleDeg 改为通用 lateral_output，支持 curvature 控制类型）
新增 LatControlCurvature 分支、steerControlType.curvature
cc.forceDecel 用 driverMonitoringState.noResponseForceDecel 替代 awarenessStatus < 0
CC.longActive 条件加 or not self.CP_SP.pcmCruiseSpeed
删除 params_thread 后台线程、ModelStateBase 继承、get_lat_delay 导入（livedelay helpers）
与 modeld 关联：直接。desiredCurvature 来源迁移到 lateralManeuverPlan；LAT_SMOOTH_SECONDS 是 modeld 常量。
迁移必要性：必须。这是 modeld 新架构下 lateral 路径消费的核心接线点。
4. selfdrive/controls/plannerd.py — 必须迁移
源 vs 当前：同路径
性质：SubMaster 订阅 modelV2，poll=modelV2 → poll=carState
关键改动：
poll 从 'modelV2' 改为 'carState'（modelV2 不再是 plannerd 的节拍源）
新增订阅 'liveMapDataSP'、'carStateSP'、gps_location_service
LongitudinalPlanner(CP) → LongitudinalPlanner(CP, CP_SP)
longitudinal_planner.sla.update_car_state(sm['carState']) 新增调用
msg.valid = sm.all_checks(['carState','carControl','modelV2','liveParameters']) → sm.all_checks()（全量校验）
与 modeld 关联：直接。poll 源从 modelV2 切走是 modeld 新架构的节拍重构信号。
迁移必要性：必须。
5. cereal/services.py — 必须迁移
源 vs 当前：同路径
性质：消息服务注册表，定义 modeld 发布的 services
关键改动：
新增 "lateralManeuverPlan": (True, 20.) —— modeld 新架构下游的新 service（lateral planner 消费）
新增 "liveLocationKalman": (True, 20.)
modelV2 队列大小 → QueueSize.BIG；modelDataV2SP → QueueSize.BIG
新增 QueueSize 枚举（BIG/MEDIUM/SMALL）整体重构
删除 gyroscope2、accelerometer2、magnetometer、lightSensor、gpsNMEA、gnssMeasurements、androidLog(→operatingSystemLog)、uploaderState、navInstruction、navRoute、navThumbnail 等
与 modeld 关联：直接。lateralManeuverPlan 是 modeld 新架构产物；modelV2/modelDataV2SP 队列扩大与 modeld 输出体积变化对应。
迁移必要性：必须（否则 lateralManeuverPlan 无法被 cereal 注册，controlsd/selfdrived 订阅会失败）。
6. selfdrive/selfdrived/selfdrived.py — 必须迁移
源 vs 当前：同路径
性质：订阅 modelV2、modelDataV2SP，新增 lateralManeuverPlan 处理
关键改动：
ignore 列表新增 'lateralManeuverPlan'
SubMaster 新增订阅 'driverMonitoringState'、'lateralManeuverPlan'、'longitudinalPlanSP'
新增 startup event：self.sm.recv_frame['lateralManeuverPlan'] > 0 → EventName.lateralManeuver
modelDataV2SP.laneTurnDirection 消费逻辑保留：custom.TurnDirection.turnLeft → 本地别名 TurnDirection（同字段，纯重构）
新增 DM（driver monitoring）lockout/事件处理（与 modeld 无关，属 selfdrived 重构）
删除 tici NVMe loggerd 忽略逻辑
radarErrors 处理块被删除（canError/radarTempUnavailable 事件移到别处）
与 modeld 关联：直接。lateralManeuverPlan 是 modeld 新消息；modelDataV2SP 消费未变但 import 路径变。
迁移必要性：必须。
7. selfdrive/selfdrived/events.py — 必须迁移（事件表）
源 vs 当前：同路径
性质：消费 driverMonitoringState，新增 lateralManeuver 事件
关键改动：
新增 EventName.lateralManeuver 事件定义（WARNING + PERMANENT）—— 对应 modeld 新架构的 lateralManeuverPlan 路径
新增 EventName.stockLkas
DM 事件名重命名：preDriverDistracted→driverDistracted1、promptDriverDistracted→driverDistracted2、driverDistracted→driverDistracted3；unresponsive 同理 1/2/3
AudibleAlert 从 car.CarControl.HUDControl.AudibleAlert → log.SelfdriveState.AudibleAlert（schema 重构）
too_distracted_alert 新函数，消费 sm['driverMonitoringState'].lockout / lockoutRecoveryPercent
canError 警报文案改为 "Unknown Vehicle Variant"
mici 设备分支追加事件覆盖
与 modeld 关联：lateralManeuver 事件直接关联 modeld 新架构；DM 事件重命名与 modeld 无直接关系但同批迁移。
迁移必要性：必须。
8. selfdrive/ui/onroad/model_renderer.py — 必须迁移
源 vs 当前：同路径
性质：UI 渲染，消费 modelV2.position.x/y/z、laneLines、roadEdges、laneLineProbs、roadEdgeStds、leadsV3（通过 radarState）
关键改动：
_update_raw_points：lane_lines / road_edges / path 的 y 坐标加 self._camera_offset（新增 CameraOffset 参数消费）—— modelV2 坐标消费方式变化
lead.status → lead.present（随 radard schema 变更）
新增 ChevronMetrics、ModelRendererSP 多继承、draw_lead_status 调用
_blend_factor 手动递增 → FirstOrderFilter 平滑
MAX_POINTS = 200 常量删除；PATH_BLEND_INCREMENT 删除
_exp_gradient 从 dict → Gradient dataclass
_map_line_to_polygon 新增 max_distance 参数
rainbow_path 新分支
与 modeld 关联：直接。modelV2 坐标消费加了 camera_offset 偏移；lead present 联动。
迁移必要性：必须（UI 与新 modeld 输出对齐）。
9. selfdrive/ui/ui_state.py — 必须迁移
源 vs 当前：同路径
性质：UI 主状态，订阅 modelV2 等
关键改动：
SubMaster 新增订阅 onroadEvents、gpsLocationExternal、carOutput、carControl、liveParameters、testJoystick、rawAudioData + self.sm_services_ext
UIStatus 新增 LAT_ONLY / LONG_ONLY
engaged 判定加 self.sm["selfdriveStateSP"].mads.enabled
删除 pyray as rl、DEFAULT_FPS、UnknownKeyName、system.hardware 导入
_params_refresh_worker 后台线程替代主线程 _update_params
light_sensor 计算：删除 ar0231 scale 6.0 分支（统一用 exposureValPercent）
与 modeld 关联：间接。modelV2 订阅保留，主要是 UI 架构重构，但与新 modeld 时代的 onroadEvents/selfdriveStateSP 对齐。
迁移必要性：必须（UI 状态机重构，否则与新 selfdriveStateSP/onroadEvents 不兼容）。
10. selfdrive/monitoring/dmonitoringd.py — 必须迁移
源 vs 当前：同路径
性质：订阅 modelV2（在 SubMaster 列表中）
关键改动：
from .helpers import DriverMonitoring → from .policy import DriverMonitoring（helpers.py 重命名为 policy.py）
新增 demo_mode 逻辑、DM.run_step(sm, demo=...)
wheelpos_learner → wheelpos_offsetter（重命名）
put_bool_nonblocking → put_bool
与 modeld 关联：弱（modelV2 仍在 SubMaster 但用法未变；主要是 DM 重构）。
迁移必要性：必须（import 路径不存在则启动失败）。
11. sunnypilot/selfdrive/controls/lib/dec/dec.py — 必须迁移
源 vs 当前：同路径
性质：消费 radarState.leadOne.status → .present
关键改动：仅 lead_one.status → lead_one.present 一行 + import 路径
与 modeld 关联：直接（lead schema 联动）。
迁移必要性：必须。
二、源仓库新增的消费者（当前仓库缺失，需评估引入）
12. sunnypilot/selfdrive/controls/lib/e2e_alerts_helper.py — 当前仓库不存在
源：openpilot/sunnypilot/selfdrive/controls/lib/e2e_alerts_helper.py（5.5KB）
性质：消费 sm['modelV2'].position.x 做 e2e alert 触发判断
与 modeld 关联：直接。是 modelV2.position.x 的新消费者。
迁移必要性：必须引入（如果当前仓库的 longitudinal_planner / controlsd_ext 调用了它——见上面 longitudinal_planner 的 is_e2e(sm) 逻辑，很可能依赖此 helper）。
13. sunnypilot/selfdrive/controls/lib/smart_cruise_control/vision_controller.py — 当前仓库不存在
源：openpilot/sunnypilot/selfdrive/controls/lib/smart_cruise_control/vision_controller.py（8.1KB）
性质：消费 sm['modelV2'].orientationRate.z 和 sm['modelV2'].velocity.x 做 Smart Cruise Control 视觉计算
与 modeld 关联：直接。是 modelV2 orientation/velocity 的新消费者。
同目录还有 tests/test_vision_controller.py、tests/test_map_controller.py（当前均缺失）
迁移必要性：必须引入（如果当前仓库的 longitudinal_planner 调用 LongitudinalPlannerSP.update_targets 走 SCC 路径——见 longitudinal_planner diff 的 LongitudinalPlannerSP.update_targets 调用）。
14. tools/lateral_maneuvers/lateral_maneuversd.py — 当前仓库不存在
源：openpilot/tools/lateral_maneuvers/lateral_maneuversd.py
性质：订阅 modelV2，发布 lateralManeuverPlan —— 这是 lateralManeuverPlan 消息的发布者
与 modeld 关联：直接。lateralManeuverPlan 是 controlsd/selfdrived 新订阅的消息，由这个 tool 进程发布。
迁移必要性：必须引入（否则 controlsd 的 sm['lateralManeuverPlan'].desiredCurvature 没有发布者，回退到 modelV2.action.desiredCurvature）。
15. selfdrive/ui/mici/onroad/* 与 selfdrive/ui/sunnypilot/mici/onroad/* — 当前仓库不存在
源：openpilot/selfdrive/ui/mici/onroad/{model_renderer.py, confidence_ball.py, ...}、.../sunnypilot/mici/onroad/{confidence_ball.py, hud_renderer.py, model_renderer.py}
性质：mici 设备 UI，消费 modelV2 做车道线/置信度绘制
与 modeld 关联：直接（modelV2 渲染消费者，针对 mici 硬件）。
迁移必要性：可选/平台相关。当前是 tici 分支；若不目标 mici 平台可不迁移，但若 UI 抽象层引用了 mici 模块则需占位。
三、纯重构/无关改动（可随批迁移但不阻塞 modeld）
16. selfdrive/controls/lib/ldw.py
仅 import 路径 from cereal → from openpilot.cereal。无 modeld 消费变化。
迁移必要性：必须（import 路径，否则 import 失败）但与 modeld 无关。
17. sunnypilot/selfdrive/controls/lib/nnlc/tests/test_nnlc.py 与 sunnypilot/selfdrive/controls/lib/dec/tests/pytest_dynamic_controller.py
import 路径 + LatControlTorque / controller.update(...) 签名变化（新增 DT_CTRL / lat_delay 参数）。
与 modeld 关联：间接（latcontrol 签名随 controlsd 重构变化）。
迁移必要性：必须（测试要跑通新签名）。
18. selfdrive/monitoring/helpers.py → policy.py（源仓库重命名）
当前仓库仍是 helpers.py，源改为 policy.py。dmonitoringd.py 和 events.py 都 import .policy。
迁移必要性：必须（重命名 + import 联动）。
四、未发现的预期消费者（任务清单中提到但实际无差异）
selfdrive/controls/lib/lateral_planner.py：两个仓库都不存在此文件。lateral 规划已并入 controlsd + lateral_maneuversd + lateralManeuverPlan 消息。任务清单第 4 项不适用。
selfdrive/controls/lib/drive_helpers.py：grep 未命中 modeld 消息（只定义 CONTROL_N / get_accel_from_plan 等工具），无 modeld 消费差异。任务清单第 2 项无需迁移。
drivingModelData 的下游消费者：除 modeld 自身（fill_model_msg.py / modeld.py）和 process_replay 测试外，无运行时消费者。drivingModelData 似乎是发布给日志/replay 用途，controlsd/plannerd/selfdrived 均不订阅。所以 drivingModelData 字段变化不影响下游。
cameraOdometry 的下游消费者：只有 locationd.py / calibrationd.py / locationd.cc / process_replay。locationd 是发布者也是订阅者（自环）。需要单独核对 locationd，但不在本次 modeld 迁移范围（modeld 已迁移，cameraOdometry 由 modeld 发布，locationd 消费——若 modeld 输出的 cameraOdometry schema 变了，locationd 要同步）。建议补查 locationd.py/locationd.cc 的 diff。
五、迁移优先级排序
优先级	文件	理由
P0	cereal/services.py	注册 lateralManeuverPlan，否则下游订阅全失败
P0	selfdrive/controls/controlsd.py	desiredCurvature 来源迁移、LAT_SMOOTH_SECONDS、lat_delay 接线
P0	selfdrive/controls/lib/longitudinal_planner.py	parse_model 删除、MPC 签名变化、present 联动
P0	selfdrive/controls/radard.py	lead status→present、lead_prob 滤波、字段删除
P0	selfdrive/controls/plannerd.py	poll 源从 modelV2→carState、CP_SP 传入
P0	tools/lateral_maneuvers/lateral_maneuversd.py	新文件，lateralManeuverPlan 发布者
P0	sunnypilot/.../e2e_alerts_helper.py	新文件，modelV2.position.x 消费者
P0	sunnypilot/.../smart_cruise_control/vision_controller.py	新文件，modelV2 orientation/velocity 消费者
P0	selfdrive/selfdrived/selfdrived.py	lateralManeuverPlan 订阅 + ignore、事件
P0	selfdrive/selfdrived/events.py	lateralManeuver 事件、DM 事件重命名
P0	selfdrive/ui/onroad/model_renderer.py	modelV2 坐标加 camera_offset、present
P0	selfdrive/ui/ui_state.py	UI 状态机重构
P1	selfdrive/monitoring/dmonitoringd.py + helpers→policy 重命名	DM 重构
P1	sunnypilot/.../dec/dec.py	present 联动（一行）
P1	sunnypilot/.../nnlc/tests/test_nnlc.py、dec/tests/pytest_dynamic_controller.py	测试签名同步
P1	selfdrive/controls/lib/ldw.py	import 路径
P2	selfdrive/ui/mici/onroad/*（mici 平台 UI）	仅 mici 平台，tici 可跳过
六、关键结论
modeld 新架构最显著的下游变化是 lateralManeuverPlan 新消息：desiredCurvature 从 modelV2.action.desiredCurvature 迁移到 lateralManeuverPlan.desiredCurvature，由新进程 lateral_maneuversd 发布。controlsd、selfdrived、events.py、services.py 必须同步。
leadOne.status → leadOne.present 是跨文件的 schema 重命名，影响 radard.py、longitudinal_planner.py、model_renderer.py、dec.py 至少 4 处。
modelV2 仍是核心订阅，但消费方式变化：longitudinal_planner 删除 parse_model（不再 interp position/velocity/acceleration 到 MPC），MPC 不再吃 modelV2 路径轨迹；新消费者 e2e_alerts_helper（position.x）和 vision_controller（orientationRate.z/velocity.x）出现。
modelDataV2SP 消费者只有 selfdrived（laneTurnDirection），未变；import 路径需迁移。
drivingModelData 无运行时消费者，只有日志/replay 用途，schema 变化不阻塞迁移。
cameraOdometry 消费者主要是 locationd，本次未深查，建议补查 locationd.py / locationd.cc 的 diff 以确认 modeld 输出 schema 兼容性。