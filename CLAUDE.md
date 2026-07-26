# CLAUDE.md — sunnypilot-pc 仓库指南

本文件描述 [MM-X/sunnypilot-pc](https://github.com/MM-X/sunnypilot-pc) 的当前状态（分支 `master-tici`，comma 三代设备 TICI 分支）。modeld 已从旧拆分模型 + C++ + OpenCL 架构迁移到新版 supercombo + 纯 tinygrad JIT 架构。

> 仓库于 commit `a04668fca`（`reinit: drop LFS and submodules, inline all content`）做过一次重置：丢弃原 git 历史、移除 LFS 过滤、把全部 7 个子模块（`tinygrad_repo`、`msgq_repo`、`opendbc_repo`、`rednose_repo`、`teleoprtc_repo`、`panda`、`sunnypilot/neural_network_data`）内联为普通跟踪目录。因此**不再有子模块指针、不再有 LFS、不再有 submodule update 覆盖风险**。下面凡涉及"子模块"处均指这些内联目录。

## 1. 仓库布局

仓库根 `/home/mx/sunnypilot-pc` 采用**根级包 + `openpilot/` 符号链接包**的双布局：

- 真实代码位于根级目录：`selfdrive/`、`system/`、`common/`、`sunnypilot/`、`cereal/`、`third_party/`、`tools/`、`msgq/`、`opendbc/`、`panda/`、`rednose/`、`tinygrad/`。
- `openpilot/` 是一个 Python 包目录，内部用符号链接指回根级包：
  - `openpilot/cereal -> ../cereal`
  - `openpilot/common -> ../common`
  - `openpilot/selfdrive -> ../selfdrive/`
  - `openpilot/system -> ../system/`
  - `openpilot/sunnypilot -> ../sunnypilot`
  - `openpilot/third_party -> ../third_party`
  - `openpilot/tools -> ../tools`
- 因此 `import openpilot.selfdrive.modeld` 与 `selfdrive/modeld/` 是**同一份文件**。
- `tinygrad_repo/`、`msgq_repo/`、`opendbc_repo/`、`rednose_repo/`、`teleoprtc_repo/` 等为各自上游仓库的**内联检出**（原为子模块，reinit 后变为普通目录，无独立 `.git`）。`tinygrad_repo/` 即 sunnypilot fork 的 tinygrad。
- `sunnypilot/modeld/` 与 `sunnypilot/modeld_v2/` 是**旧架构**（拆分模型 + C++ + OpenCL），由 `modeld_snpe` / `modeld_tinygrad` NativeProcess 注册（见 §4，当前已被注释禁用），与 `selfdrive/modeld/` 的新架构并存但独立，不在本指南范围。

## 2. 构建系统

- 顶层 `SConstruct` 用 SCons 递归调用各包的 `SConscript`：`common/`、`cereal/`、`panda/`、`rednose/`、`system/{ubloxd,loggerd,logcatd,proclogd,camerad}`、`third_party/`、`selfdrive/`（递归到 `selfdrive/modeld/SConscript`）、`sunnypilot/`。
- `pyproject.toml` 声明运行时依赖，含 `onnx >= 1.14.0`；pytest 配置忽略 `openpilot/`、`tinygrad_repo/`、各 `*_repo/`、`*.onnx` 等子目录（reinit 后这些虽是普通目录，但路径名仍保留 `_repo` 后缀）。
- `selfdrive/modeld/SConscript`（新版，~150 行）要点：
  - 引入 `from openpilot.selfdrive.modeld.constants import ModelConstants` 与 `get_model_metadata.py`。
  - `probe_devices()`：子进程跑 `tinygrad.Device.get_available_devices()` 探测可用后端，避免 usbgpu 锁残留。
  - `tg_backend` 探测链：`MODELD_TG_BACKEND` 环境变量覆盖 → `CUDA` → `QCOM`（`IMAGE=1 FLOAT16=1 NOLOCALS=1 JIT_BATCH_SIZE=0 OPENPILOT_HACKS=1`）→ `CL`（Mali：`IMAGE=2 FLOAT16=1 NOLOCALS=1`，注释称比 `IMAGE=1` 快）→ `CPU`（Darwin `DEV=CPU`，其他 `DEV=CPU:LLVM`）。
  - `tg_devices` 字典写入 `models/tg_input_devices.json`，键为 `selfdrive.modeld.modeld` / `selfdrive.modeld.dmonitoringmodeld`，值为 `{WARP_DEV, QUEUE_DEV}` / `{DEV}`。
  - 构建产物：`driving_tinygrad.pkl`（分块）、`dmonitoring_model_metadata.pkl`、各分辨率 `dm_warp_{W}x{H}_tinygrad.pkl`、`dmonitoring_model_tinygrad.pkl`（走 `tinygrad_repo/examples/openpilot/compile3.py`）。
  - USBGPU 串行化：`models/.usb_gpu.lock` 作为 `SideEffect`。
  - `.gitignore` 覆盖 `selfdrive/modeld/models/*.pkl*` 与 `tg_input_devices.json`，pkl 不入版本库（分块 pkl 由 `file_chunker` 生成，本地构建）。
- **RKNN 路径搁置**：`selfdrive/modeld/compile_rknn.py` 把 `driving_supercombo.onnx` 编译为 `driving_supercombo.rknn`（RK3588 NPU，FP16），但 SConscript 不引用它，运行时未启用。整网编译在 transformer 的 GatherND/bool 节点受阻止步（见 memory `rknn-supercombo-infeasible`）；重拾只能拆分 vision→NPU / policy→CPU。

## 3. 当前 modeld 架构（新：纯 tinygrad JIT + supercombo）

`selfdrive/modeld/` 现状：

- **纯 tinygrad JIT，无 C++ / 无 OpenCL / 无 runners**：已移除 `commonmodel*`、`transforms/`、`runners/`。
- **编译脚本**：
  - `compile_modeld.py`：从 `driving_supercombo.onnx` 生成融合的 `driving_tinygrad.pkl`（含 warp + policy JIT、metadata、output_slices），导出 `make_input_queues`、`WARP_INPUTS`、`POLICY_INPUTS` 供运行时调用。内含 JIT 确定性自检（seed=42，pickle 往返，`expect_match` True/False）。
  - `compile_dm_warp.py`：按相机分辨率生成 `dm_warp_{W}x{H}_tinygrad.pkl`。
  - `get_model_metadata.py`：从 ONNX 提取 metadata dict，被 `compile_modeld.py` 和 SConscript 调用。
  - `helpers.py`：`dump_oob`/`load_oob`（pickle 协议 5 带外缓冲，避免大 pkl 内存峰值）、`usbgpu_present()`、`modeld_pkl_path()`、`get_tg_input_devices()`、`TG_INPUT_DEVICES_PATH`。
- **常量与解析**（reinit 后从 fill_model_msg 拆出，供编译期与运行时共享）：
  - `constants.py`：`ModelConstants`（时间/距离索引、输入输出宽度、MHP 选择等）+ output slices（`Plan`、`Meta`）。`MODEL_RUN_FREQ = 10 if RK else 20`。
  - `parse_model_outputs.py`：`Parser`、`sigmoid`、`safe_exp` 等，把 supercombo 原始输出张量切片为语义字段。被 `modeld.py`、`dmonitoringmodeld.py`、`fill_model_msg.py` 引用。
- **模型（融合式）**：`driving_supercombo.onnx`（~61 MB）、`big_driving_supercombo.onnx`（占位 135 B，USBGPU 路径）、`dmonitoring_model.onnx`。
- **大 pkl 分块**：经 `common.file_chunker` 分块/重组（绕过 git 单文件 100 MB 限制）。运行时 `load_oob(open_file_chunked(modeld_pkl_path(...)))` 加载。
- **运行时**：`modeld.py` 继承 `sunnypilot.modeld_v2.modeld_base.ModelStateBase`，用 `Tensor.from_blob` 直接引用 visionipc 缓冲（不拷贝主机数据），warp + policy 走 tinygrad JIT。
- **跨包依赖**（迁移时补齐，现已齐全）：`common/file_chunker.py`、`system/camerad/cameras/nv12_info.{py,h}`、`sunnypilot/modeld_v2/modeld_base.py`、`sunnypilot/models/helpers.py:plan_x_idxs_helper`。
- **输出字段说明**：`selfdrive/modeld/MODEL_OUTPUT_DIFF.md` 记录 supercombo 输出张量到 cereal 字段的映射与变更。

## 4. 运行时入口与注册

- `modeld.py`：`PROCESS_NAME="selfdrive.modeld.modeld"`，发布 `modelV2`、`drivingModelData`、`cameraOdometry`、`modelDataV2SP`。继承 `ModelStateBase`，加载 driving pkl（warp + policy JIT）。
- `dmonitoringmodeld.py`：`PROCESS_NAME="selfdrive.modeld.dmonitoringmodeld"`，驾驶员监控模型。
- `system/manager/process_config.py`：
  - `PythonProcess("modeld", "selfdrive.modeld.modeld", and_(only_onroad, is_stock_model))`。
  - 旧架构 `modeld_snpe` / `modeld_tinygrad` 的 NativeProcess 注册已被注释禁用。
- **TICI 上 dmonitoring 禁用**（commit `9bdbcf77b` "hack dm"，现为 TICI 既定行为）：
  - `process_config.py` 注释掉 `dmonitoringmodeld` 与 `dmonitoringd` 的注册。
  - `selfdrived.py` 从 `camera_packets` 删 `driverCameraState`，从 SubMaster 订阅删 `driverMonitoringState`，`ignored_processes` 加 `dmonitoringmodeld`/`dmonitoringd`，注释掉 `self.events.add_from_msg(self.sm['driverMonitoringState'].events)`。
  - 原因：新 `dmonitoringmodeld` 在 TICI 上不跑。

## 5. tinygrad ops_cl Mali 补丁

reinit 后 `tinygrad_repo/` 是普通跟踪目录，`tinygrad/runtime/ops_cl.py` 的两处 Mali 补丁已作为常规内容纳入版本库（不再是"子模块内未提交修改"）。补丁仍是 CL 后端在 Mali 上运行的必要修复：

1. **`CLProgram.__call__`**：对 `ImageDType` 参数每次调用 `clCreateImage` 创建 image wrapper（NV12 frame 走 `image2d_from_buffer`），原版从不释放。Mali 为每个 image 分配 device-side descriptor，每帧 ~3.2 MB 累积 → 30 分钟 OOM。修复：`clEnqueueNDRangeKernel` 后 `for img in imgs: cl.clReleaseMemObject(img)`（enqueue 会 retain 到 kernel 完成，可安全提前 release）。
2. **`CLAllocator._alloc`**：原版忽略 `options.external_ptr`，只 `clCreateBuffer(CL_MEM_READ_WRITE, None)`，导致 `Tensor.from_blob` 在 CL 上静默丢主机指针 → warp 输入垃圾 → 轨迹不显示。修复：`external_ptr` 非空时优先 `clImportMemoryARM`（`cl_arm_import_memory` 扩展，经 `clGetExtensionFunctionAddress(b"clImportMemoryARM")` 取符号，常量 `CL_IMPORT_TYPE_ARM=0x40B2`、`CL_IMPORT_TYPE_HOST_ARM=0x40B3`，`cl_import_properties_arm` 为 `intptr_t` 64 位），fallback `CL_MEM_USE_HOST_PTR`（数据正确但 Mali 有 driver shadow 开销）。

⚠️ **覆盖风险变化**：不再有 `git submodule update` 覆盖问题。但如果未来重新从上游同步 `tinygrad_repo/`（如 `git pull` 上游或重新内联），这两处补丁会被上游版本冲掉。丢失后症状：`tg_backend='CL'` 时 modeld 30 分钟内 OOM 或轨迹不显示；`tg_backend='CPU'` 正常。恢复方法见 memory `tinygrad-cl-fromblob-fix`，或直接重新应用上述两处补丁。

## 6. 下游消费者自洽结论

modeld 迁移后，**下游消费者未跟随源仓库演进**，当前仓库内部自洽：

- `selfdrive/controls/controlsd.py:142` 仍从 `model_v2.action.desiredCurvature` 读，当前 `modeld.py` / `fill_model_msg.py` 仍发布该字段。
- `selfdrive/controls/radard.py:259` 仍用 `leadOne`（当前仓库内部 RadarState schema，非 modeld 输出）。
- `cereal/services.py` 无 `lateralManeuverPlan`，当前不使用源仓库的新 lateral 链路。
- `selfdrive/modeld/fill_model_msg.py` 填充的全是当前 cereal 标准 `modelV2`/`cameraOdometry`/`drivingModelData` 字段，未引用源仓库新消息。

源仓库 `sunnypilot-pc-new` 有一大批消费者改动（`lateralManeuverPlan` 新链路、`leadOne.status→present` schema 重命名、UI 状态机重构、DM 事件重命名、MPC 签名重构等），属**整体版本演进**，非 modeld 迁移适配所必需，**未迁移**。

备忘：若未来要同步源仓库演进，需**整批**迁移（不能只迁一两个文件，会破坏 schema 自洽）：`cereal/services.py` + `selfdrive/controls/{controlsd,radard,plannerd}.py` + `selfdrive/controls/lib/longitudinal_planner.py` + `selfdrive/selfdrived/{selfdrived,events}.py` + `selfdrive/ui/onroad/model_renderer.py` + `selfdrive/ui/ui_state.py` + 新文件 `tools/lateral_maneuvers/lateral_maneuversd.py` + `sunnypilot/selfdrive/controls/lib/e2e_alerts_helper.py` + `sunnypilot/selfdrive/controls/lib/smart_cruise_control/vision_controller.py` 等。

## 7. 本地回归验证（离线数据）

process_replay / model_replay 默认从远端 Azure 拉取测试 route，本地开发可改用 vendored 数据离线回归：

- `tools/lib/local_route.py`：`local_route_path(route, sidx, fn)`，在 `LOCAL_ROUTE_DIR` 下按 `<route>--<sidx>/<fn>` 或 `<log_id>--<sidx>/<fn>`（`tools/replay/data` 的目录命名）解析本地文件，找不到返回 `None` 供调用方回退远端。
- 设 `LOCAL_ROUTE_DIR=tools/replay/data` 后，下列脚本本地优先、找不到才走远端（CI 不设此 env，仍用 Azure）：
  - `selfdrive/test/process_replay/test_processes.py`：`segments` 缩为单条 VW demo route（`a2a0ccea32023010|00000004--9a1ce93c08--0`）；`full_test` 多车断言与 `--update-refs` full-test 要求在本地模式跳过。原 17 车列表保留在注释中。
  - `selfdrive/test/process_replay/model_replay.py`：`TEST_ROUTE`/`SEGMENT` 切到 demo route seg 0（可用 `MODEL_REPLAY_ROUTE` / `MODEL_REPLAY_SEGMENT` env 覆盖）；首次无本地 ref 时把当前输出存为基线到 `fakedata/` 并退出 0，再跑才对比。
  - `selfdrive/test/process_replay/regen.py` 与 `test_regen.py`：rlog/hevc 走 `local_route_path` 优先。
- demo route 无 driver cam，本地模式用 fcamera 作 dummy dcam（与 regen 的 `dummy_driver_cam` 一致）。
- 相关 env：`LOCAL_ROUTE_DIR`、`MODEL_REPLAY_ROUTE`、`MODEL_REPLAY_SEGMENT`、`FILEREADER_CACHE`。
- 详见 `LOGGING_AND_VERIFICATION.md` 第四章。

## 8. 常用命令

- 构建：`scons -j8`（根目录）
- 仅 modeld：`scons selfdrive/modeld`
- 运行时环境：`source launch_env.sh`
- 启动 manager（跳过固件查询）：`SKIP_FW_QUERY=1 python3 system/manager/manager.py`
- 启动：`./launch_openpilot.sh`
- 本地离线回归示例：
  - `LOCAL_ROUTE_DIR=tools/replay/data python3 selfdrive/test/process_replay/test_processes.py --whitelist-cars VOLKSWAGEN --whitelist-procs controlsd`
  - `LOCAL_ROUTE_DIR=tools/replay/data python3 selfdrive/test/process_replay/model_replay.py`
