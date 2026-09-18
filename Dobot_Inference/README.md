# Dobot_Inference

这是新版9/9维位置控制数据的π0.5推理逻辑：

- 输入：6关节实测角、夹爪实测位置、底盘实测线/角速度，共9维；
- 图像：Top 相机和腕部 RealSense；
- 输出：6关节目标角、夹爪连续目标位置、底盘目标线/角速度，共9维；
- 升降台固定在 0.107 m，不进入该检查点的输入和输出；
- 20 Hz 策略动作块、50 Hz 关节控制，新旧动作块进行平滑融合；
- 默认只做 DRY-RUN，不使能任何执行器。

旧 `dataset_meta/` 仅作为26K历史快照，不再被新版推理读取。新版默认从
`datasets/lerobot/nova_pi05_mobile_position_v3/`读取元数据，也可以通过
`NOVA_DATASET_ROOT`指定；权重通过`NOVA_CHECKPOINT`指定。旧11维权重不能用于新版。

## 准备环境

```bash
cd /home/nvidia/dsw_ws/nova_vla/lerobot-main
uv sync --locked
uv pip install --python .venv/bin/python -r ../requirements.txt
```

先确认代码和控制逻辑测试通过：

```bash
cd /home/nvidia/dsw_ws/nova_vla
lerobot-main/.venv/bin/python -m unittest Dobot_Inference.tests.test_control -v
```

## 启动相机

`run_teleop.sh` 会自动启动相机，但推理不应同时运行遥操。推理前单独启动 Top 相机：

```bash
cd /home/nvidia/dsw_ws/nova_vla
/home/nvidia/miniforge3/envs/single/bin/python -m Dobot_Init.top_camera --no-preview
```

保持该终端运行，并用下面命令检查：

```bash
curl -fsS http://127.0.0.1:8080/health
```

## 先做 DRY-RUN

新终端执行：

```bash
cd /home/nvidia/dsw_ws/nova_vla
NOVA_CHECKPOINT=/实际/检查点目录 bash run_inference.sh
```

DRY-RUN 会读取双相机和真机状态、加载权重并持续打印模型动作，但不会使能机械臂、
夹爪或底盘。必须确认日志显示严格完整加载权重，而且 9 维动作顺序正确。

## 真机执行

只有在 DRY-RUN 正常、急停可用且工作区无人时运行：

```bash
cd /home/nvidia/dsw_ws/nova_vla
NOVA_CHECKPOINT=/实际/检查点目录 \
NOVA_EXECUTE=1 \
NOVA_RUN_SECONDS=180 \
bash run_inference.sh
```

程序会再次要求现场按 Enter。任何异常都会先给底盘发送零速度。模型会常驻 GPU，
一轮结束后直接按 Enter 可复用权重开始下一轮，输入 `q` 才释放模型。

## 版本说明

这个 26K 检查点最初由 LeRobot 0.5.1 训练；当前工程中的 `lerobot-main` 是 0.6.1。
推理入口会检查权重是否严格完整加载，若没有出现完整加载标记会直接拒绝真机运行。
在 NVIDIA 上第一次迁移必须先做 DRY-RUN；如果 0.6.1 的处理器配置不兼容，就应使用
训练该权重时保存的 0.5.1 源码环境，不能绕过检查。
