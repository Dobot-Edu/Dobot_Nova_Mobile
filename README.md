# Nova VLA 移动操作工程

这是 Nova 机械臂、夹爪、固定升降台、移动底盘和 Quest VR 的统一工程。配置集中在
`config/`，运行代码按功能分目录，数据和运行产物不混入源码。

```text
nova_vla/
├── config/          # 设备、任务、采集和训练参数的唯一配置入口
├── Dobot_Init/           # 机械臂、夹爪、升降台、底盘和相机底层接口
├── Dobot_Collection/     # Quest 遥操与 25 Hz 原始数据采集
├── Dobot_Train/          # LeRobot 源码和加载助手
├── Dobot_Inference/      # π0.5 双相机真机推理与控制
├── quest_vr/             # Quest APK、通信协议、坐标映射和运动学
├── scripts/              # 数据、回零、标定、相机工具；diagnostics为专项诊断
├── system/               # 系统服务模板
├── tests/
│   ├── pipeline/         # 数据流水线测试
│   └── teleop/           # 遥操模块单元测试
├── datasets/
│   ├── raw/              # 原始 episode
│   ├── lerobot/          # 转换后的 LeRobot v3 数据集
│   └── excluded/         # 人工排除的 episode
├── run_teleop.sh         # 启动 Top 相机和遥操
├── run_inference.sh      # 启动 π0.5 推理（默认真机执行）
```

`logs/` 和 `outputs/` 会在运行时自动生成，不属于源码。此干净工程不包含历史快照、
旧项目副本和已经采集的数据。

相机启动后默认在主机桌面弹出可缩放的Top/Wrist双画面窗口：每路640×480，目标20FPS（config/中的CAMERA_PREVIEW_FPS），实际刷新率取决于渲染负载。硬件采流保持30FPS，原始数据按25Hz保存。
遥操在单击A初始化后弹出；推理在模型加载完成、相机启动后弹出。
窗口共享已有图像，不重复连接相机；关闭窗口不停止控制或采集。
SSH启动会尝试定位同一用户的桌面会话；主机未登录桌面或缺少Tk时打印提示，主程序继续。
在 `config/` 设置 `CAMERA_PREVIEW_ENABLED = True/False` 开启/关闭双相机预览，
修改后重启程序生效；不再读取 `NOVA_PREVIEW` 环境变量。
请勿同时运行会独占腕部相机的旧 `scripts.view_cameras` 查看器。

## 1. 修改配置

普通使用只修改 `config/`。常用项目如下：

- `TELEOP_PROFILE`：`full`、`arm_only` 或 `test`；`full` 固定升降台；
- `TASK`：写入数据集的任务文字；
- `RELABEL_TASK_DURING_CONVERSION`：把旧 episode 的语言指令统一重标为当前 `TASK`；
- `VIDEO_CODEC`：视频编码器；NVIDIA 上用 `auto` 自动选择硬件 H.264；
- `DATASET_FPS`：采集频率；统一入口没有时长或条数上限；
- `RAW_DATASET_DIR`、`LEROBOT_DATASET_DIR`：原始与转换后数据路径；
- `STARTUP_JOINT_DEG`、`STARTUP_SPEED_FACTOR`：机械臂标准位与回位速度；
- `TRAIN_STEPS`、`BATCH_SIZE`、`POLICY_CHUNK_SIZE`：训练参数。

## 2. 上传到 NVIDIA

本地工程路径：`/home/yfy/dsw_ws/nova_vla`。

从笔记本执行：

```bash
rsync -av --info=progress2 \
  --exclude 'datasets/raw/' \
  --exclude 'datasets/lerobot/' \
  --exclude 'datasets/excluded/' \
  --exclude 'logs/' \
  --exclude 'outputs/' \
  /home/yfy/dsw_ws/nova_vla/ \
  nvidia@10.2.150.205:/home/nvidia/dsw_ws/nova_vla/
```

如需迁移 NVIDIA 旧工程中的原始数据，在 NVIDIA 上执行：

```bash
mkdir -p /home/nvidia/dsw_ws/nova_vla/datasets/raw
rsync -av --info=progress2 \
  /home/nvidia/project/nova_pi05_mobile/data/raw/ \
  /home/nvidia/dsw_ws/nova_vla/datasets/raw/
```

## A/B 遥操与数采统一入口（2026-09-10）

```bash
cd /home/nvidia/dsw_ws/nova_vla
source run_teleop.sh full       # 或 arm_only / test；bash 运行同样支持
```

启动后处于待机：不自动清错、使能、张开夹爪或回位。现场确认路径清空、急停可用后，
单击 A 并松开；约0.3秒的双击判定窗口结束后，程序执行自检、张开夹爪，并以机械臂
当前实测位姿作为遥操相对零点，不自动回标准位。需要回标准位时再双击 A。

机械臂 ServoJ 被拒绝时，暂停机械臂指令并尝试点亮红灯，底盘遥操继续可用。
排除碰撞后双击 A：先停止底盘，再尝试一次状态恢复和回标准位；失败则保持机械臂暂停，
手柄回中后仍可操作底盘。控制柜报警期间可能拒绝灯光 DO 指令，终端会打印该错误。
数据遥测带有 `arm_fault` 诊断字段；发生碰撞的录制需要人工检查，不应直接当成成功示范。
启动恢复会等待禁用确认为 Mode 4，再使能并连续确认 Mode 5，避免 JOG 退出的瞬态状态。
等待“就绪”提示；初始化/回位后必须先松开 Grip、摇杆回中、松开 A/B，旧输入不会直接恢复。

| 模式/操作 | 行为 |
|---|---|
| full / arm_only，单击 A | 首次初始化并激活；激活后单击不改变零点、不关闭遥操 |
| full / arm_only，单击 B | ready 时立即开录；recording 时停录并保存，不回位，遥操继续 |
| full / arm_only，双击 A | 停底盘，机械臂回初始标准位；录制不中断 |
| full / arm_only，长按 A/B、AB 同按 | 不分配功能；双击 B 无独立功能，约0.3秒防连击 |

Grip 按住跟随机械臂，松开保持；重新握住仍建立相对零点。A 原来的重设零点逻辑已删除。
Trigger 按住逐步闭合、松开逐步张开，保留现有编码器反馈与限力逻辑。
full 摇杆控制底盘，无需 Grip；arm_only 摇杆禁用，但底盘仍须通电并提供真实反馈。
full/arm_only 不允许升降台运动。

初始化后自动创建独立采集进程。看到 `[数采 ready]` 后才能按 B；没有倒计时。
未激活、初始化/回位中、采集准备中或保存中按 B 不排队开录。正常按 B 结束的条目全部保存，
metadata.success=true 仅表示默认纳入人工筛选，未做任务成功审核；转换前手动删除不需要的整条目录。
无时长上限，无条数自动退出，无终端 Enter/y/n。保存完成后回到 ready，等待下一次 B。
Ctrl+C、进程异常或数据源超时导致未完成的条目保留 `.partial`，不进入转换。

机械臂控制柜使能后，实体信号灯自动显示状态：DO8黄灯表示遥操已激活但未录制，
DO7绿灯表示正在采集，DO6红灯表示遥操或数采异常。正常 Ctrl+C 退出时三个灯全部熄灭。
因为控制柜未使能时不能控制信号灯，所以启动后、尚未单击 A 初始化的待机阶段不亮灯。

数采已经整合进 `run_teleop.sh`，不要再启动第二份采集进程。
采集子进程拥有腕部相机和17890端口，不要同时运行独立相机/遥测占用工具。
当前9/9维数据、底盘命令/反馈和夹爪反馈均按25Hz采集配置运行。

## 5. 检查并转换数据

转换前必须先检查原始数据：

```bash
cd /home/nvidia/dsw_ws/nova_vla
/home/nvidia/miniforge3/envs/single/bin/python -m scripts.validate_raw
```

当前转换器只接受新版9维状态/9维动作，并要求 metadata 明确声明夹爪 `1=张开、0=闭合`。
旧维度或旧夹爪语义会被拒绝，不能与本轮数据混合。任务文字可按
`RELABEL_TASK_DURING_CONVERSION` 重标为当前 `TASK`，图像和时序不会被改写。

确认检查结果后，直接运行 v3 转换器，不需要额外 shell 包装脚本：

```bash
/home/nvidia/miniforge3/envs/single/bin/python -m scripts.convert_to_lerobot_v3
```

输出默认为 `datasets/lerobot/nova_pi05_mobile_position_v3/`。转换器拒绝覆盖已有输出；需要
重新转换时，修改 `config/` 中的 `LEROBOT_DATASET_DIR` 和
`DATASET_REPO_ID`，或先人工确认并移走旧输出。

当前schema为9维状态、9维动作。夹爪状态使用25Hz编码器实测位置，夹爪动作使用连续
目标位置，均为1=张开、0=闭合。升降台物理固定在0.107m，不进入模型输入和输出；`full`
遥操也会禁用升降台按键。旧26K的11维检查点与新版输入不兼容。

## 6. 训练

训练服务器安装本仓库的 LeRobot 和 π0.5 训练依赖后，使用原生 `lerobot-train`。
`config/training.py` 的 `TRAIN_ARGUMENTS` 可交给 AI 生成命令。
生成时按训练服务器上的实际位置填写 `dataset.root` 和 `output_dir`。

启动前检查数据集为 LeRobot v3，状态和动作均为 9 维，且 `meta/stats.json` 含两者的 `q01` / `q99`。
新训练选择尚未存在的输出目录。

## 7. 相机、测试和真机诊断

π0.5 推理的权重准备、DRY-RUN 和真机命令见
[`Dobot_Inference/README.md`](Dobot_Inference/README.md)。第一次必须先运行 DRY-RUN，
不要同时运行遥操程序。

双相机查看器：

```bash
/home/nvidia/miniforge3/envs/single/bin/python -m scripts.view_cameras
```

不连接真机的测试：

```bash
python -m pytest -q scripts/tests
```

常用标定/回零工具在 `scripts/`，专项诊断在 `scripts/diagnostics/`。确认现场条件后
才运行，例如：

```bash
/home/nvidia/miniforge3/envs/single/bin/python -m scripts.lift_home --execute
/home/nvidia/miniforge3/envs/single/bin/python -m scripts.gripper_calibrate --status
/home/nvidia/miniforge3/envs/single/bin/python -m scripts.gripper_calibrate --execute
/home/nvidia/miniforge3/envs/single/bin/python -m scripts.diagnostics.lift_diagnostic
/home/nvidia/miniforge3/envs/single/bin/python -m scripts.diagnostics.chassis_motion_test --help
```

夹爪标定中 `--status` 完全只读，不切换力矩或发送位置；`--execute` 才允许小步点动。
动作端点和编码器反馈端点分别配置，但二者对上层都统一为 `0=安全全开、1=安全全闭`。
禁止将机械极限位置直接作为动作端点。

转换使用本工程 `Dobot_Train/lerobot-main/src`，训练机安装同一份源码后调用 `lerobot-train`。
转换只接受 `schema_version=2` 的9维状态/9维动作，不再裁剪旧维度或反转旧夹爪语义。
默认视频编码为 CPU H.264；`python -m unittest scripts.tests.pipeline.test_pipeline` 会测试
MP4写入和LeRobot回读。旧夹爪电流诊断已停用；系统服务模板在 `system/`，尚未重新安装服务。

### 配置文件

| 文件 | 参数 |
| --- | --- |
| `config/devices.py` | Quest 网络、设备连接和相机 |
| `config/teleop.py` | 遥操模式、坐标映射和跳变阈值 |
| `config/paths.py` | 项目和数据路径 |
| `config/dataset.py` | 数采、数据校验与转换 |
| `config/training.py` | π0.5 训练 |
| `config/inference.py` | π0.5 推理 |
| `config/runtime.env` | 启动脚本默认环境 |
