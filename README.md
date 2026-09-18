# Nova VLA 移动操作工程

Nova 机械臂、夹爪、固定升降台、移动底盘和 Quest VR 的统一工程。

日常配置分为两处：

- `constants.py`：遥操、数采、数据转换和训练参数。
- `Dobot_Inference/settings.py`：π0.5 权重、提示词、推理频率和真机执行参数。

## 1. 环境配置

<details>
<summary>环境配置（待补充）</summary>

TODO：补充 NVIDIA 主机初始化、Python 环境、设备权限、Quest 网络、相机和训练服务器环境。

当前 NVIDIA 运行环境默认使用：

```bash
/home/nvidia/miniforge3/envs/single/bin/python
```

训练机需要另外安装项目对应版本的 LeRobot 和 π0.5 训练依赖。

</details>

## 2. 遥操方法

启动后程序处于待机状态。确认工作区无障碍、急停可用后，单击右手柄 A 键初始化。初始化会执行机械臂自检、按配置清错和使能、张开夹爪，并以机械臂当前位置作为遥操相对零点；不会自动回到标准位。

### full / arm_only

| 手柄输入 | 功能 |
|---|---|
| 单击 A | 首次初始化并激活遥操；激活后再次单击不执行操作 |
| 双击 A | 底盘停车，机械臂回到标准初始位；正在进行的数采不会停止 |
| 单击 B | ready 状态开始数采；recording 状态结束并保存 |
| 按住 Grip | 机械臂跟随右手柄运动 |
| 松开 Grip | 机械臂保持当前位置 |
| 按住 Trigger | 夹爪连续闭合 |
| 松开 Trigger | 夹爪连续张开 |
| 右摇杆前 / 后 | 底盘前进 / 后退，仅 `full` 生效 |
| 右摇杆左 / 右 | 底盘左转 / 右转，仅 `full` 生效 |

`arm_only` 不允许摇杆驱动底盘，但仍会读取真实底盘反馈。长按 A、长按 B、双击 B 均无功能；A/B 同按不会触发操作。

### no_chassis 维护模式

| 手柄输入 | 功能 |
|---|---|
| 单击 A | 首次初始化并激活维护遥操 |
| 长按 A | 升降台上升 |
| 长按 B | 升降台下降 |
| A/B 同按或松开 | 升降台停止 |
| Grip / Trigger | 机械臂和夹爪操作与其他模式相同 |

维护模式不启动数采，也不连接底盘。

实体灯状态：

- 黄灯：遥操已激活，未录制。
- 绿灯：正在采集。
- 红灯：遥操或数采异常。

## 3. 程序运行命令

以下命令均在项目根目录执行：


### 遥操与数采

```bash
source run_teleop.sh full
```

可将 `full` 改为 `arm_only` 或 `no_chassis`。数采已经集成在遥操程序中，不需要单独启动采集脚本。按 B 开始，再按 B 结束并保存。

### 数据检查与转换

先检查原始数据：

```bash
python -m scripts.validate_raw
```

检查通过后转换为 LeRobot v3：

```bash
python -m scripts.convert_to_lerobot_v3
```

原始数据目录、输出目录、任务提示词以及转换时是否覆盖提示词均在 `constants.py` 配置。

### 训练

```bash
python -m Dobot_Train.check_training_setup
bash run_training.sh
```

训练参数在 `constants.py` 中配置。训练机的 Python 不是 `python3` 时：

### π0.5 推理

先在 `Dobot_Inference/settings.py` 中确认权重路径、数据集元信息、提示词和执行参数。

默认 DRY-RUN，不会使能执行器：

```bash
bash run_inference.sh
```

也可以临时指定权重：

```bash
NOVA_CHECKPOINT=/path/to/checkpoint bash run_inference.sh
```

DRY-RUN 检查正常后再启动真机执行：

```bash
NOVA_EXECUTE=1 bash run_inference.sh
```

## 4. 常用命令

### 升降台回零

这是实机运动命令，执行后还需要在终端输入 `HOME`：

```bash
/home/nvidia/miniforge3/envs/single/bin/python -m scripts.lift_home --execute
```

### 机械臂清除报错

保持 `constants.py` 中 `ROBOT_AUTO_CLEAR_ERROR = True`，启动遥操后单击 A，初始化流程会查询报警、执行一次 `ClearError` 并使能机械臂，不下发运动目标。

如果遥操过程中机械臂进入故障暂停，排除碰撞后双击 A 会尝试恢复并回到标准位。该操作会产生机械臂运动。

### 夹爪只读检查

只读取编码器位置、电流、温度和运动状态，不发送位置命令：

```bash
/home/nvidia/miniforge3/envs/single/bin/python -m scripts.gripper_calibrate --status
```

夹爪端点标定会产生实机运动，只有需要重新标定时才运行：

```bash
/home/nvidia/miniforge3/envs/single/bin/python -m scripts.gripper_calibrate --execute
```

### 查看相机

```bash
/home/nvidia/miniforge3/envs/single/bin/python -m scripts.view_cameras
```

### 检查或复现最新一条数据

默认只检查数据，不连接硬件：

```bash
/home/nvidia/miniforge3/envs/single/bin/python -m scripts.replay_episode latest
```

真机复现会要求输入 `REPLAY`：

```bash
/home/nvidia/miniforge3/envs/single/bin/python -m scripts.replay_episode latest --execute
```

## 5. 注意事项

- 遥操、Replay 和推理不能同时运行，避免多个进程争用机械臂、夹爪、底盘和相机。
- π0.5 推理默认是 DRY-RUN；第一次加载新权重必须先确认模型完整加载、9 维动作顺序和输出趋势。
- 单击 A 初始化会清错、使能并张开夹爪；双击 A 和升降台回零命令会产生真实运动。
- B 结束的 episode 会直接保存，`metadata.success=true` 只表示默认纳入转换，不代表任务已经人工验收。
- 转换前先运行 `scripts.validate_raw`，并人工移走失败或不需要的完整 episode。
- 转换器不会覆盖已有输出目录；重新转换时需要修改 `LEROBOT_DATASET_DIR` 和 `DATASET_REPO_ID`，或人工移走旧输出。
- 当前数据为 9 维 observation、9 维 action，夹爪在数据集中统一为 `1=张开，0=闭合`。
- 当前数采为 25 Hz，机械臂控制为 50 Hz，相机流为 30 Hz；π0.5 推理频率由 `Dobot_Inference/settings.py` 配置。
- 升降台固定在 0.107 m，不进入模型 observation 或 action。
- `constants.py` 的改动影响遥操、数采、转换和训练；`Dobot_Inference/settings.py` 的改动只影响 π0.5 推理。
