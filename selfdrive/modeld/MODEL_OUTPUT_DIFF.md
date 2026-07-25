# modeld 迁移前后输出差异

迁移提交 `0a361a6fd`（[TICI] modeld: migrate to supercombo + pure tinygrad architecture）。对比 `0a361a6fd^`（旧：拆分模型 + C++ + OpenCL）与当前（新：supercombo + 纯 tinygrad JIT）。

## A. 消息层面（modeld 发布到 cereal 的内容）

### 1. 发布的消息集合 — 不变

`PubMaster(["modelV2", "drivingModelData", "cameraOdometry", "modelDataV2SP"])`，`PROCESS_NAME="selfdrive.modeld.modeld"` 保持。四个消息新旧一致。

### 2. `modelV2` 字段填充 — 完全相同

`fill_model_msg` 填充的字段集合不变：

| 区段 | 字段 |
|---|---|
| header | `frameId` `frameIdExtra` `frameAge` `frameDropPerc` `timestampEof` `modelExecutionTime` `valid` |
| plan | `position` `velocity` `acceleration` `orientation` `orientationRate`（各 t/x/y/z + Std） |
| action | `action`（desiredCurvature / desiredAcceleration / shouldStop） |
| laneLines(4) | `t` `x` `y` `z`；`laneLineStds` `laneLineProbs` |
| roadEdges(2) | `t` `x` `y` `z`；`roadEdgeStds` |
| leadsV3(3) | `t` `x` `y` `v` `a` + Std；`prob` `probTime` |
| meta | `desireState` `desirePrediction` `engagedProb` `disengagePredictions`（t/brakeDisengageProbs/gasDisengageProbs/steerOverrideProbs/brake3MetersPerSecondSquaredProbs/brake4MetersPerSecondSquaredProbs/brake5MetersPerSecondSquaredProbs/gasPressProbs/brakePressProbs）`hardBrakePredicted` |
| confidence | `confidence`（green/yellow/red） |
| raw | `rawPredictions`（仅 `SEND_RAW_PRED` 时） |

### 3. `cameraOdometry` 字段 — 完全相同

`frameId` `timestampEof` `trans` `rot` `wideFromDeviceEuler` `roadTransformTrans` + 各 Std。

### 4. `laneLines`/`roadEdges` 的 `t` 字段 — **实质差异**

| | 旧 | 新 |
|---|---|---|
| `LINE_T_IDXS` | `[]`（空，注释"aren't used"） | `plan_x_idxs_helper(ModelConstants, Plan, net_output_data)`（按 plan 的 X 坐标采样，非空） |

旧版车道线/路沿的 `t` 字段填空列表；新版填实际时间点。下游若消费 `laneLines[i].t` 会观察到值变化。

### 5. `drivingModelData` 填充方式 — **结构性差异**

| | 旧 | 新 |
|---|---|---|
| `fill_model_msg` 签名 | `(base_msg, extended_msg, ...)` 同时填 drivingModelData + modelV2 | `(msg, ...)` 只填 modelV2 |
| drivingModelData 来源 | 独立填充（`action`/`path`/`laneLineMeta`/`frameId` 等直接写） | 新函数 `fill_driving_model_data(msg, modelv2_send)` 从 modelV2 **复制** |
| 一致性保证 | 两消息独立写，字段可能不一致 | drivingModelData 是 modelV2 的子集副本，保证一致 |

新版 `fill_driving_model_data` 从 `modelV2` 复制：`frameId`/`frameIdExtra`/`frameDropPerc`/`modelExecutionTime`/`action`/`meta.laneChangeState`/`meta.laneChangeDirection`/`laneLineMeta`/`path`（poly 拟合）。

### 6. import 路径 — 变更

| | 旧 | 新 |
|---|---|---|
| cereal | `from cereal import log` | `from openpilot.cereal import log` |
| helpers | — | `from openpilot.sunnypilot.models.helpers import plan_x_idxs_helper` |

## B. 模型内部输出（supercombo 融合 vs 拆分）

### 7. 模型架构 — 根本差异

| | 旧 | 新 |
|---|---|---|
| ONNX | `driving_vision.onnx`（视觉编码）+ `driving_policy.onnx`（policy RNN）两个 | `driving_supercombo.onnx` 单一融合模型 |
| 推理 | 两阶段：vision → policy，policy 带 hidden state 递推 | 单阶段：warp + vision + policy 融合，一次出全部输出 |
| warp | Python 侧 OpenCL transforms（`transforms/transform.cl` + `loadyuv.cl`）做 YUV→模型输入变换 | warp 作为 tinygrad JIT（`self.warp`）内嵌进 supercombo 计算图，`Tensor.from_blob` 直接引用 visionipc NV12 缓冲 |
| metadata | `driving_vision_metadata.pkl` + `driving_policy_metadata.pkl` 两套 | 单一 `metadata`（含 `input_shapes`/`output_slices`/warp+policy JIT） |

### 8. `action` 来源 — 新增直接输出分支

| | 旧 | 新 |
|---|---|---|
| `get_action_from_model` | 总是从 plan 计算（`get_accel_from_plan` + `get_curvature_from_plan`） | `if 'action' not in model_output:` 回退到从 plan 计算；**else** 直接用 `model_output['action'][0,0]`（曲率）/`[0,1]`（加速度） |

supercombo 可直接输出 action；但发布到 `modelV2.action` 的字段新旧一致。

### 9. hidden_state 递推 — 机制相同，实现不同

| | 旧 | 新 |
|---|---|---|
| RNN 状态 | policy_inputs 含 `desire`/`prev_desired_curv`/`traffic_convention`/`action_t`；policy_output 含 hidden state | `self.npy['prev_feat'][:] = model_output[self.output_slices['hidden_state']]`；supercombo 输出 `hidden_state` 供下一帧输入 |

### 10. 常量 — 一致

`Plan`/`Meta` 枚举（POSITION/VELOCITY/ACCELERATION/ORIENTATION_RATE/T_FROM_CURRENT_EULER/ENGAGED/BRAKE_DISENGAGE/GAS_DISENGAGE/STEER_OVERRIDE/HARD_BRAKE_3/4/5/GAS_PRESS/BRAKE_PRESS）新旧一致。`ModelConstants` 的 `T_IDXS`/`X_IDXS`/`LEAD_T_IDXS`/`DESIRE_LEN` 等一致。

## C. 下游影响

- `controlsd.py`、`radard.py`、`plannerd.py`、`ui` 消费的 `modelV2` 字段未变，**无需适配**（见 CLAUDE.md §6）。
- 唯一可观察差异：`laneLines[i].t` 从空 → 有值。当前仓库无消费者读 `laneLines.t`（UI 用 `laneLines.y`），无影响。
- `drivingModelData` 现为 `modelV2` 子集副本，字段一致性更强。
