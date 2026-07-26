# 日志与回归验证手段总览

本仓库（`master-tici` 分支）的日志、诊断、回归验证手段梳理。所有路径相对仓库根 `/home/mx/sunnypilot-pc`。

---

## 一、日志 / 日志记录手段

### 1. swaglog —— 进程级文本日志（核心）

**实现**：[common/swaglog.py](common/swaglog.py)、[common/swaglog.cc](common/swaglog.cc)（C++ 侧桥接）、[common/logging_extra.py](common/logging_extra.py)（`SwagLogger`/`SwagFormatter`/`SwagLogFileFormatter`）。

**机制**：
- 每个进程 `import` `cloudlog`（即 `SwagLogger` 单例）。日志同时走两条出口：
  1. **stdout `StreamHandler`**：受 `LOGPRINT` 环境变量控制（`debug`/`info`/`warning`，默认 `warning`）。
  2. **`UnixDomainSocketHandler`**：把 record 序列化（首字符 = levelno）经 ZMQ PUSH 发到 `ipc:///tmp/logmessage`（[Paths.swaglog_ipc()](system/hardware/hw.py#L35)）。**不直接落盘**，避免磁盘 I/O 阻塞主流程；满了 `zmq.NOBLOCK` 直接丢弃。
- `logmessaged` 守护进程（[system/logmessaged.py](system/logmessaged.py)）PULL 这个 socket，做三件事：
  - level ≥ INFO 写入滚动文件 `swaglog.NNNNNNNNNN`（`SwaglogRotatingFileHandler`：每 60s 或 256KB 滚一次，保留 2500 份）。
  - 转发为 cereal `logMessage` 消息发布。
  - level ≥ ERROR 再发一条 `errorLogMessage`。
- `add_file_handler()`：当 `logmessaged` 未运行时（如 PC 调试），可在进程内直接挂文件 handler 兜底。

**落盘位置**：`Paths.swaglog_root()` —— 设备 `/data/log/`，PC `~/.comma/log/`（[system/hardware/hw.py:28](system/hardware/hw.py#L28)）。文件名 `swaglog.0000000001` 等。

**记录内容**：进程名、PID、时间戳、level、调用栈定位（文件:行号）、自定义消息。`SwagLogFileFormatter` 为文件格式，`SwagFormatter` 为 IPC/控制台格式。

**读取方式**：直接 `cat`/`tail` 文件；或订阅 `logMessage`/`errorLogMessage` 消息（plotjuggler、replay 可看）。`LOGPRINT=debug` 提升控制台输出。

**`cloudlog.*` 与 `cloudlogbomb`**：`cloudlog.debug/info/warning/error`；`logbomb` 用于一次性大块诊断 dump。

---

### 2. loggerd —— 行驶数据 capnp 日志（rlog/qlog）

**实现**：[system/loggerd/](system/loggerd/)（C++ 主体 `logger.cc`/`loggerd.cc` + Python `config.py`/`deleter.py`/`uploader.py`/`xattr_cache.py`）。

**机制**：
- 订阅 `cereal/services.py` 列出的全部服务消息，按 `logMonoTime` 顺序写入。
- 每个 segment 一个目录：`<dongle_id>|<route>--<sidx>/`，内含：
  - `rlog.zst` —— **完整**消息流（Zstd 压缩，`ZstdFileWriter`，[logger.cc:190](system/loggerd/logger.cc#L190)）。
  - `qlog.zst` —— **精简**流（`in_qlog=true` 的消息，用于上传/预览，[logger.cc:199-201](system/loggerd/logger.cc#L199)）。
  - 视频文件 `fcamera.hevc`/`dcamera.hevc`/`ecamera.hevc`/`qcamera.mp4`（由 `encoderd` 编码）。
- `init_data` 在 segment 开头写一次（设备指纹、版本、carParams 等）。
- `bootlog`（[system/loggerd/bootlog.cc](system/loggerd/bootlog.cc)）启动时记录一次 boot 标记。

**落盘位置**：`Paths.log_root()` —— 设备 `/data/media/0/realdata/`，PC `~/.comma/media/0/realdata/`，可被 `LOG_ROOT` 环境变量覆盖（[hw.py:15](system/hardware/hw.py#L15)）。外挂存储 `/mnt/external_realdata/`。

**上传/清理**：`uploader.py` 后台上传 qlog+视频到 comma ai；`deleter.py` 在磁盘不足时按最旧优先删除。`xattr_cache.py` 用扩展属性记录上传状态。

**读取方式**：
```python
from openpilot.tools.lib.logreader import LogReader
lr = LogReader("/path/to/rlog.zst")   # 或远端 route
for m in lr: print(m.which())         # capnp DynamicStructReader
```
`rlog.zst` 也可被 `tools/replay/replay` 二进制直接回放。

---

### 3. logcatd —— Android logcat 采集

**实现**：[system/logcatd/](system/logcatd/)（`logcatd_systemd.cc`）。抓 Android 系统 logcat，发布为 cereal `androidLog` 消息并进 rlog。TICI 上不跑。

---

### 4. proclogd —— 进程/系统资源采集

**实现**：[system/proclogd/](system/proclogd/)（`proclog.cc`/`main.cc`）。周期性采集 `/proc` 下的进程列表、CPU/内存、文件描述符等，发布为 cereal `procLog` 消息并进 rlog。用于事后分析内存/CPU 回归、找泄漏。

**读取**：LogReader 读 `procLog` 字段。

---

### 5. ubloxd —— GPS 原始观测

**实现**：[system/ubloxd/](system/ubloxd/)。解析 u-blox 串口原始报文，发布 `ubloxGnss`/`gpsLocationExternal` 等，进 rlog。回归验证里 `ubloxd` 是被测进程之一。

---

### 6. tinygrad 运行时计数器（modeld 调试）

**实现**：`tinygrad_repo/tinygrad/engine/` 的 `GlobalCounters`（jit 调用次数、kernel 数、`mem_used` 等）。

**用法**：在 modeld 进程里 `from tinygrad.engine.jit import TinyJit`；调试时可打印 `GlobalCounters`。`modeld.py` 自身主要用 `cloudlog.debug` 打 vipc 帧异常（[selfdrive/modeld/modeld.py:225](selfdrive/modeld/modeld.py#L225)）。CLAUDE.md §5 提到曾用 `GlobalCounters.mem_used` 与 RSS 对比定位 tinygrad 统计之外的泄漏。

**注意**：CL 后端有 `ops_cl.py` 两处未提交修复（见 CLAUDE.md §5、memory `tinygrad-cl-fromblob-fix`），丢失会导致 `tg_backend='CL'` 时 30 分钟 OOM 或轨迹不显示——靠 swaglog + RSS 监控定位。

---

### 7. crash / 异常根

**路径**：`Paths.crash_log_root()` —— 设备 `/data/community/crashes/`，PC `~/.comma/community/crashes/`（[hw.py:79](system/hardware/hw.py#L79)）。进程崩溃时落 tombstone/trace。`logmessaged` 把 ≥ERROR 的转 `errorLogMessage`。

---

### 8. 路径速查（[system/hardware/hw.py](system/hardware/hw.py)）

| Paths 方法 | 设备 | PC | 环境变量覆盖 |
|---|---|---|---|
| `log_root()` | `/data/media/0/realdata/` | `~/.comma/media/0/realdata` | `LOG_ROOT` |
| `swaglog_root()` | `/data/log/` | `~/.comma/log` | — |
| `swaglog_ipc()` | `ipc:///tmp/logmessage` | 同（加 `OPENPILOT_PREFIX`） | — |
| `download_cache_root()` | `/tmp/comma_download_cache/...` | 同 | `COMMA_CACHE` |
| `crash_log_root()` | `/data/community/crashes/` | `~/.comma/community/crashes` | — |
| `model_root()` | `/data/media/0/models/` | `~/.comma/media/0/models` | — |

`OPENPILOT_PREFIX` 给所有 comma 路径加后缀，用于同机多实例隔离。

---

## 二、回归验证手段

### 1. process_replay —— 进程级输出回归（主力）

**入口**：[selfdrive/test/process_replay/](selfdrive/test/process_replay/)，README [selfdrive/test/process_replay/README.md](selfdrive/test/process_replay/README.md)，驱动脚本 [test_processes.py](selfdrive/test/process_replay/test_processes.py)。

**原理**：取一条已知 route 的 rlog 作为输入，喂给单个进程重跑（不经 manager），把输出与"参考日志"逐字段比较；字段不一致即回归。每个车厂一条代表 segment。

**被测进程**：controlsd / radard / plannerd / calibrationd / dmonitoringd / locationd / paramsd / ubloxd / torqued / **modeld** / **dmonitoringmodeld**。

**关键文件**：
- [process_replay.py](selfdrive/test/process_replay/process_replay.py)：`replay_process()` / `replay_process_with_name()` / `get_process_config()` / `get_custom_params_from_lr()`。
- [compare_logs.py](selfdrive/test/process_replay/compare_logs.py)：`compare_logs()` 用 `dictdiffer` 做字段级 diff，支持 `ignore_fields`/`ignore_msgs`/`tolerance`，`format_diff()` 输出。
- [regen.py](selfdrive/test/process_replay/regen.py)：`regen_segment()` 把整条 route 重新跑所有进程生成新日志。
- [regen_all.py](selfdrive/test/process_replay/regen_all.py)：并行多 segment 批量 regen。
- [ref_commit](selfdrive/test/process_replay/ref_commit)：参考日志对应的 commit。
- [migration.py](selfdrive/test/process_replay/migration.py)：参考日志 schema 迁移。

**依赖**：rlog.zst 数据（远端 comma ci 或本地 `FILEREADER_CACHE=1` 缓存）、`modeld`/`dmonitoringmodeld` 还需 `FrameReader` 喂相机帧。

**使用**：
```bash
# 跑全部进程回归（默认对比远端参考日志）
cd selfdrive/test/process_replay && ./test_processes.py

# 缓存 log 文件到本地
FILEREADER_CACHE=1 ./test_processes.py

# 只测 controlsd + HONDA
./test_processes.py --whitelist-procs controlsd --whitelist-cars HONDA

# 忽略某些字段
./test_processes.py --ignore-fields driverMonitoringState.events

# 用当前 commit 重生成参考日志（intentional 变更后）
./test_processes.py --update-refs
```

**程序化用法**（README 给出）：
```python
from openpilot.selfdrive.test.process_replay import replay_process_with_name
from openpilot.tools.lib.logreader import LogReader
lr = LogReader(...)
out = replay_process_with_name('locationd', lr)

# modeld 需要 frs
from openpilot.tools.lib.framereader import FrameReader
frs = {'roadCameraState': FrameReader(...), 'wideRoadCameraState': FrameReader(...)}
out = replay_process_with_name(['modeld','dmonitoringmodeld'], lr, frs=frs)
```

**fork 自有参考日志**：fork 默认本地存参考日志。`./test_processes.py` 生成 → git-lfs 提交 → 更新 `ref_commit` 文件。本仓库即此模式。

---

### 2. model_replay —— 模型推理回归（modeld 专用）

**入口**：[selfdrive/test/process_replay/model_replay.py](selfdrive/test/process_replay/model_replay.py)。

**原理**：固定测试 route `8494c69d3c710e81|000001d4--2648a9a404`，segment 4，frame 0–60。跑 `modeld`/`dmonitoringmodeld`，输出与 master 参考日志对比；同时检查执行耗时上限（`modelV2` 35ms 即时/25ms 均值；`driverStateV2` 20ms/15ms）。

**产物**：matplotlib PNG 对比图（`plot()`，MASTER vs PROPOSED 曲线），结果上传到 `model_replay_master` bucket 并由 `GithubUtils` 在 PR 上贴评论。

**依赖**：`CI_ARTIFACTS_TOKEN`、`GITHUB_COMMENTS_TOKEN`、远端 route 的 fcamera/eCamera hevc（FrameReader）。

**触发**：CI `model_review` workflow（见下）或手动 `python model_replay.py`。

---

### 3. 模型 ONNX ↔ tinygrad JIT 自检（编译期）

**入口**：[selfdrive/modeld/compile_modeld.py](selfdrive/modeld/compile_modeld.py) 的 `compile_jit()`（[L259](selfdrive/modeld/compile_modeld.py#L259)）。

**原理**：JIT 捕获后用固定 seed=42 随机输入跑一次，存输出 `val` 和输入 buffer；然后 pickle 往返（`dump_oob`/`load_oob`）再跑同 seed → assert 输出**完全相等**（`expect_match=True`）；换 seed=43 → assert 输出**不等**（`expect_match=False`，确认没被意外缓存/常量折叠）。打印每次 enqueue/total 耗时。

**依赖**：`tinygrad.nn.onnx.OnnxRunner`、`driving_supercombo.onnx`、`common.file_chunker`（分块 pkl 重组）。

**使用**：
```bash
python selfdrive/modeld/compile_modeld.py \
  --model-size 256x512 \
  --camera-resolutions 1928x1208 \
  --onnx selfdrive/modeld/models/driving_supercombo.onnx \
  --output driving_tinygrad.pkl \
  --frame-skip 4
```
失败说明 JIT capture/pickle/确定性有 bug。**注意**：这是确定性自检，不是"与 ONNX 数值对比"——数值正确性靠 process_replay 的 modeld 回归（金标准 rlog）兜底。

---

### 4. test_modeld —— modeld 端到端冒烟测试

**入口**：[selfdrive/modeld/tests/test_modeld.py](selfdrive/modeld/tests/test_modeld.py)（pytest）。

**原理**：起 `VisionIpcServer` 喂全零 NV12 帧给 `managed_processes['modeld']`，订阅 `modelV2`/`cameraOdometry`，验证 frameId 对齐、消息正常产生。不发真车数据，纯"能跑起来+输出对齐"冒烟。

**使用**：`pytest selfdrive/modeld/tests/test_modeld.py`（CI 用 `pytest -n logical`）。

---

### 5. replay —— 全栈回放（人在回路）

**入口**：[tools/replay/](tools/replay/)，README [tools/replay/README.md](tools/replay/README.md)，二进制 `tools/replay/replay`（aarch64 ELF，C++），封装脚本 [tools/replay/replay.sh](tools/replay/replay.sh)。

**原理**：读 rlog + hevc，按时间戳把 `roadCameraState`/`can`/`pandaStates`/`peripheralState` 等回灌 msgq/visionipc，同时跑真实 `manager.py`（controlsd/modeld/...），UI 可视化。和 process_replay 区别：这是**全栈在线**回放（人在回路调试），process_replay 是**离线单进程**逐字段对比。

**Fillback 模式**（[replay.sh](tools/replay/replay.sh)）：`SKIP_FW_QUERY=1 FILLBACK=1`，跳过 panda 固件查询、`selfdrived` 放宽 liveness，replay 自带 roadCameraState/can。固定 allow-list `roadEncodeIdx,roadCameraState,can,pandaStates,peripheralState`，其余（modelV2/controlState...）由运行栈自己产生。

**使用**：
```bash
./tools/replay/replay.sh --demo                                   # 内置 demo route（无需 auth）
./tools/replay/replay.sh 'a2a0ccea32023010|2023-07-27--13-01-19' --data_dir=./tools/replay/data
tools/replay/replay <route> --dcam --ecam                         # watch3 三摄
ZMQ=1 tools/replay/replay <route>                                 # 用 ZMQ 代替 MSGQ
# 配合 plotjuggler：
tools/plotjuggler/juggle.py --stream
# 配合 UI：
cd selfdrive/ui && ./ui
```
`--allow`/`--block` 白黑名单、`-c` 缓存 segment 数、`-s` 起始秒、`-x` 倍速、`--no-vipc` 不出视频。

**fixture log**：`tools/replay/data/` 下内置 `*.zst` fixture route（commit `8b41c7de0` "track *.zst fixture logs"），`--demo` 即用此，免远端认证。

**can_replay**：[tools/replay/can_replay.py](tools/replay/can_replay.py) 把 rlog 的 CAN 包经 `PandaJungle` 灌进真实 CAN 总线（硬件在环），支持上下电/点火循环（`PWR_ON/OFF`、`ON/OFF`）。

---

### 6. regen —— 旧 segment 重生成新 segment

**入口**：[regen.py](selfdrive/test/process_replay/regen.py) / [regen_all.py](selfdrive/test/process_replay/regen_all.py) / [test_regen.py](selfdrive/test/process_replay/test_regen.py)。

**原理**：把一条老 route 整条过 process_replay 生成"新版本"输出日志，作为后续回归参考或验证 openpilot 能否 engage（`check_openpilot_enabled`）。`test_regen.py` 在 CI 上对几条代表性 segment 跑，断言能 engage。

---

### 7. CI workflows（[.github/workflows/](.github/workflows/)）

| Workflow | 作用 |
|---|---|
| [selfdrive_tests.yaml](.github/workflows/selfdrive_tests.yaml) | 主 CI：`pytest -n logical`，`FILEREADER_CACHE=1`，docker `sunnypilot-tici-base` 镜像，跑全部 `test_*.py`（含 process_replay、test_modeld） |
| [model_review.yaml](.github/workflows/model_review.yaml) | PR 改 `selfdrive/modeld/models/*.onnx` 时触发，`scripts/reporter.py` 对比 master 的 onnx 输出，贴 PR 评论 |
| [build-all-tinygrad-models.yaml](.github/workflows/build-all-tinygrad-models.yaml) | 手动触发，重编译全部 tinygrad pkl 模型并推送，设最小 selector 版本 |
| [build-single-tinygrad-model.yaml](.github/workflows/build-single-tinygrad-model.yaml) | 单模型编译 |
| [cereal_validation.yaml](.github/workflows/cereal_validation.yaml) | cereal schema 校验 |
| [sunnypilot-build-model.yaml](.github/workflows/sunnypilot-build-model.yaml) | sunnypilot 模型构建 |

**pytest 配置**（[pyproject.toml](pyproject.toml)）：`addopts` 忽略 `openpilot/`、`opendbc/`、`panda/`、`tinygrad_repo/` 等子目录；`-Werror --strict-config --strict-markers --durations=10 -n auto --dist=loadgroup`；marker `slow`/`tici`/`skip_tici_setup`；`testpaths` 含 `common/selfdrive/system/...`。`cpp_harness = selfdrive/test/cpp_harness.py` 支持 C++ 测试。

**单模块跑法**：
```bash
pytest selfdrive/modeld/tests/test_modeld.py -v
pytest selfdrive/test/process_replay/test_processes.py --whitelist-procs controlsd
pytest common/tests/ -k test_params
pytest -m "not slow"                  # 跳过慢测
pytest -m tici                        # 只跑 TICI 专属
```

---

## 三、快速决策表

| 想做的事 | 用哪个 |
|---|---|
| 看进程报错 | `~/.comma/log/swaglog.*` 或订阅 `errorLogMessage` |
| 看行驶数据 | LogReader 读 `rlog.zst` |
| 分析内存/CPU 回归 | rlog 里 `procLog` |
| 调试单进程输出差异 | `test_processes.py --whitelist-procs X` |
| 验证 modeld 输出没回归 | process_replay 的 modeld / `model_replay.py` |
| 验证 JIT 编译正确 | `compile_modeld.py` 自检（seed 确定性） |
| 冒烟验证 modeld 能起 | `test_modeld.py` |
| 人在回路全栈调试 | `tools/replay/replay.sh --demo` |
| 硬件在环 CAN 回放 | `tools/replay/can_replay.py` |
| PR 改了 onnx | CI `model_review` 自动评论 |

---

## 四、本地（离线）数据运行

默认 process_replay / model_replay 从 Azure `openpilotci` 拉输入 rlog + hevc + 参考日志。设 `LOCAL_ROUTE_DIR` 后，所有数据拉取**先查本地目录**，命中即用本地、未命中回退远端（CI 不设此变量，行为不变）。本地语料只 vendored 了一条 demo route（VW Golf，`a2a0ccea32023010|00000004--9a1ce93c08`，3 segment，在 [tools/replay/data/](tools/replay/data/)），多车覆盖被有意收窄。

### 本地解析器

[tools/lib/local_route.py](tools/lib/local_route.py)：`local_route_path(route, sidx, fn)` 在 `LOCAL_ROUTE_DIR` 下按 `<route>--<sidx>/<fn>` 或 `<log_id>--<sidx>/<fn>`（replay 的目录约定，只用 `|` 后部分）找文件。demo route 无 dcamera，`model_replay` / `regen` 的 `--dummy-dcamera` 用 fcamera 顶替。

### 跑法

```bash
# 前置：进 venv + 设环境
source .venv/bin/activate
export PYTHONPATH=$(pwd)
export LOCAL_ROUTE_DIR=$(pwd)/tools/replay/data

# 1. 单进程回归（断网 OK，输入走本地 demo route）
cd selfdrive/test/process_replay
./test_processes.py --whitelist-cars VOLKSWAGEN --whitelist-procs controlsd

# 2. 首次生成 demo route 的参考日志进 fakedata/（本地基线）
./test_processes.py --whitelist-cars VOLKSWAGEN --update-refs --local
# 之后手动：改 ref_commit 文件为当前 commit，git add fakedata/*.zst

# 3. 模型回归（首次无 ref → 自动存基线到 fakedata/，再跑才对比）
python selfdrive/test/process_replay/model_replay.py

# 4. regen 单 segment（不engage 也跑得起来）
python selfdrive/test/process_replay/regen.py a2a0ccea32023010|00000004--9a1ce93c08 0 --dummy-dcamera

# 5. regen engage 测试（LOCAL_ROUTE_DIR 下自动切 demo route）
pytest selfdrive/test/process_replay/test_regen.py -v
```

### 覆盖范围与限制

- **只覆盖 VW Golf**：controlsd/radard/plannerd/calibrationd/locationd/paramsd/ubloxd/torqued/modeld 的回归只在这一个车厂上验；车型特定路径（其他车厂接口）验不到。要恢复多车覆盖，把 [test_processes.py](selfdrive/test/process_replay/test_processes.py) 顶部 `segments` 改回 17 条 regen 列表（注释里保留了原始 17 车名）、不设 `LOCAL_ROUTE_DIR`。
- **modeld 回归**用 demo route seg 0（有 wideRoadCameraState + eCamera，能跑）。`driverStateV2` 用 fcam 做 dummy dcam，输出可对比但不是真实驾驶员视角。
- **model_replay 首次**：`fakedata/{route}_model_tici_master.zst` 不存在时，脚本把本次输出存为基线并提示 "Re-run to compare"，`sys.exit(0)`；第二次起才对比。
- **CI 不受影响**：CI 不设 `LOCAL_ROUTE_DIR`，仍走 Azure + `FILEREADER_CACHE=1`。
- **car 覆盖断言**：`full_test` 的多车厂断言和 `--update-refs` 的 full-test 要求在 `LOCAL_ROUTE_DIR` 设置时自动跳过。

### 相关 env

| 变量 | 作用 |
|---|---|
| `LOCAL_ROUTE_DIR` | 本地 route 根目录（设了才启用本地优先；不设=远端，CI 默认） |
| `MODEL_REPLAY_ROUTE` | 覆盖 model_replay 的测试 route（默认 demo route） |
| `MODEL_REPLAY_SEGMENT` | 覆盖 segment 号（默认 0） |
| `FILEREADER_CACHE` | `=1` 时缓存远端 log 到本地（CI 用，本地用不到） |
