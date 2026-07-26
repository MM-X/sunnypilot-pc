![](https://user-images.githubusercontent.com/47793918/233812617-beab2e71-57b9-479e-8bff-c3931347ca40.png)

---

# sunnypilot-pc

## 🌞 这是什么？

[sunnypilot-pc](https://github.com/MM-X/sunnypilot-pc) 是 [sunnypilot](https://github.com/sunnyhaibin/sunnypilot)（comma.ai [openpilot](https://github.com/commaai/openpilot) 的 fork）的个人定制 fork，本仓库默认开发分支为 `master-tici`，面向 comma 三代设备（TICI / comma 3X / RK3588 / PC）。

相对上游的主要差异：
- **TICI(black panda)支持**：老的comma3支持, 非canfd。
- **RK3588硬件支持**：CL后端使用Mali加速推理，模型推理频率10HZ，适配RK硬件800x420显示分辨率, 无需外接imu，使用can上yawRate数据（检查车型can的yawRate是否有值）。
- **PC支持**： 可自选NVIDIA/APPLE M系列/CPU加速，模型推理频率10HZ，无需外接imu，使用can上yawRate数据（检查车型can的yawRate是否有值）， 需要自行适配相机内参（参考仓库其他分支）。
- **最新模型迁移**：移除了旧拆分模型 + C++ + OpenCL 链路，运行时无 C++、无 OpenCL runners。warp + policy 走 tinygrad JIT。

仓库内部布局、构建系统、modeld 架构、tinygrad 补丁、运行时入口等的完整说明见 [`CLAUDE.md`](CLAUDE.md)。

## 🚘 在车上运行

- 一台支持的设备：comma 3X / comma three（本分支目标硬件为 TICI）。
- 本软件。
- 一台[支持的车型](docs/CARS.md)（继承自 openpilot，含 Honda、Toyota、Hyundai、Nissan、Kia、Chrysler、Lexus、Acura、Audi、VW、Ford 等）。车辆需具备自适应巡航与车道保持辅助。
- 一根 [car harness](https://comma.ai/shop/products/car-harness) 连接车辆。

设备在车内的安装参考 [comma 安装指南](https://comma.ai/setup)。

## 🛠 构建与运行

```bash
# 构建（根目录）
scons -j8
# 仅构建 modeld
scons selfdrive/modeld

# 运行时环境
source launch_env.sh

# 启动 manager（跳过固件查询）
SKIP_FW_QUERY=1 python3 system/manager/manager.py

# 启动
./launch_openpilot.sh
```

## 🧪 本地回归验证

process_replay / model_replay 默认从远端拉取测试 route，本地开发可改用 vendored 数据离线回归（设 `LOCAL_ROUTE_DIR=tools/replay/data`）：

```bash
LOCAL_ROUTE_DIR=tools/replay/data python3 selfdrive/test/process_replay/test_processes.py --whitelist-cars VOLKSWAGEN --whitelist-procs controlsd
LOCAL_ROUTE_DIR=tools/replay/data python3 selfdrive/test/process_replay/model_replay.py
```

日志机制与回归验证手段的完整说明见 [`LOGGING_AND_VERIFICATION.md`](LOGGING_AND_VERIFICATION.md)。

## 🎆 贡献

欢迎通过 GitHub 提交 Pull Request 与 Issue。Bug 修复尤为欢迎。PR 请针对 `master` 分支。

## 📊 用户数据

默认情况下，sunnypilot 会将驾驶数据上传至 comma 服务器。你也可以通过 [comma connect](https://connect.comma.ai/) 访问自己的数据。

sunnypilot 是开源软件，用户可自行禁用数据采集。

sunnypilot 记录前视摄像头、CAN、GPS、IMU、磁力计、热传感器、崩溃与操作系统日志。驾驶员摄像头与麦克风仅在设置中明确开启时才记录。

使用本软件即表示你了解：使用本软件或其相关服务会生成特定用户数据，这些数据可能由 comma 自行决定记录与存储。接受此协议即授予 comma 对该数据不可撤销、永久、全球范围的使用权。

## 许可证

sunnypilot 基于 [MIT License](LICENSE) 发布。本仓库包含原创工作，以及大量源自 [comma.ai 的 openpilot](https://github.com/commaai/openpilot) 的代码，后者同样以 MIT 许可发布并附带额外免责声明。原 openpilot 许可声明（含 comma.ai 赔偿条款与 alpha 软件免责声明）如下：

> openpilot is released under the MIT license. Some parts of the software are released under other licenses as specified.
>
> Any user of this software shall indemnify and hold harmless Comma.ai, Inc. and its directors, officers, employees, agents, stockholders, affiliates, subcontractors and customers from and against all allegations, claims, actions, suits, demands, damages, liabilities, obligations, losses, settlements, judgments, costs and expenses (including without limitation attorneys’ fees and costs) which arise out of, relate to or result from any use of this software by user.
>
> **THIS IS ALPHA QUALITY SOFTWARE FOR RESEARCH PURPOSES ONLY. THIS IS NOT A PRODUCT.
> YOU ARE RESPONSIBLE FOR COMPLYING WITH LOCAL LAWS AND REGULATIONS.
> NO WARRANTY EXPRESSED OR IMPLIED.**

完整许可条款见 [`LICENSE`](LICENSE)。

---

# sunnypilot-pc (English)

## 🌞 What is this?

[sunnypilot-pc](https://github.com/MM-X/sunnypilot-pc) is a personal fork of [sunnypilot](https://github.com/sunnyhaibin/sunnypilot) (itself a fork of comma.ai's [openpilot](https://github.com/commaai/openpilot)), an open source driver assistance system. The default development branch is `master-tici`, targeting comma three-generation hardware (TICI / comma 3X / RK3588 / PC).

Notable differences from upstream:
- **TICI (black panda) support**: supports the older comma 3, non-CAN-FD.
- **RK3588 hardware support**: the CL backend accelerates inference via Mali at a 10 Hz model inference rate, adapts to the RK hardware's 800x420 display resolution, needs no external IMU, and uses yawRate from CAN (check that the car's CAN yawRate reports a value).
- **PC support**: choose NVIDIA / Apple M-series / CPU acceleration, 10 Hz model inference rate, no external IMU needed, uses yawRate from CAN (check that the car's CAN yawRate reports a value), and you must adapt the camera intrinsics yourself (see other branches in this repo).
- **Latest model migration**: removed the old split-model + C++ + OpenCL path — no C++ runtime, no OpenCL runners. warp + policy run via tinygrad JIT.

For the full repo layout, build system, modeld architecture, tinygrad patches, and runtime entry points, see [`CLAUDE.md`](CLAUDE.md).

## 🚘 Running on a dedicated device in a car

- A supported device: comma 3X / comma three (this branch targets TICI).
- This software.
- One of [the supported cars](docs/CARS.md) (inherited from openpilot — Honda, Toyota, Hyundai, Nissan, Kia, Chrysler, Lexus, Acura, Audi, VW, Ford and more). The car must have adaptive cruise control and lane-keeping assist.
- A [car harness](https://comma.ai/shop/products/car-harness) to connect to the car.

Instructions for [how to mount the device in a car](https://comma.ai/setup).

## 🛠 Build & Run

```bash
# Build (repo root)
scons -j8
# Build modeld only
scons selfdrive/modeld

# Runtime environment
source launch_env.sh

# Launch manager (skip firmware query)
SKIP_FW_QUERY=1 python3 system/manager/manager.py

# Launch
./launch_openpilot.sh
```

## 🧪 Local regression verification

process_replay / model_replay pull test routes from a remote source by default; for local development you can run them offline against vendored data by setting `LOCAL_ROUTE_DIR=tools/replay/data`:

```bash
LOCAL_ROUTE_DIR=tools/replay/data python3 selfdrive/test/process_replay/test_processes.py --whitelist-cars VOLKSWAGEN --whitelist-procs controlsd
LOCAL_ROUTE_DIR=tools/replay/data python3 selfdrive/test/process_replay/model_replay.py
```

Full documentation of logging mechanisms and regression verification tooling is in [`LOGGING_AND_VERIFICATION.md`](LOGGING_AND_VERIFICATION.md).

## 🎆 Pull Requests

Both pull requests and issues are welcome on GitHub. Bug fixes are encouraged. Pull requests should be against the `master` branch.

## 📊 User Data

By default, sunnypilot uploads driving data to comma servers. You can also access your data through [comma connect](https://connect.comma.ai/).

sunnypilot is open source software. The user is free to disable data collection.

sunnypilot logs the road-facing camera, CAN, GPS, IMU, magnetometer, thermal sensors, crashes, and operating system logs. The driver-facing camera and microphone are only logged if you explicitly opt in in settings.

By using this software, you understand that use of this software or its related services will generate certain types of user data, which may be logged and stored at the sole discretion of comma. By accepting this agreement, you grant comma an irrevocable, perpetual, worldwide right to use this data.

## Licensing

sunnypilot is released under the [MIT License](LICENSE). This repository includes original work as well as significant portions of code derived from [openpilot by comma.ai](https://github.com/commaai/openpilot), which is also released under the MIT license with additional disclaimers. The original openpilot license notice, including comma.ai's indemnification and alpha software disclaimer, is reproduced below as required:

> openpilot is released under the MIT license. Some parts of the software are released under other licenses as specified.
>
> Any user of this software shall indemnify and hold harmless Comma.ai, Inc. and its directors, officers, employees, agents, stockholders, affiliates, subcontractors and customers from and against all allegations, claims, actions, suits, demands, damages, liabilities, obligations, losses, settlements, judgments, costs and expenses (including without limitation attorneys’ fees and costs) which arise out of, relate to or result from any use of this software by user.
>
> **THIS IS ALPHA QUALITY SOFTWARE FOR RESEARCH PURPOSES ONLY. THIS IS NOT A PRODUCT.
> YOU ARE RESPONSIBLE FOR COMPLYING WITH LOCAL LAWS AND REGULATIONS.
> NO WARRANTY EXPRESSED OR IMPLIED.**

For full license terms, see the [`LICENSE`](LICENSE) file.
