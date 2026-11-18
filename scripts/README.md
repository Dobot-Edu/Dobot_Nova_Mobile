# scripts

这里保存手动运行的数据、查看、标定和诊断工具；自动测试在 `tests/`，系统服务在根目录 `system/`。

- `validate_raw.py`：逐条检查原始 episode、图像、向量和时间戳；
- `convert_to_lerobot_v3.py`：转换为 Parquet、MP4 和 LeRobot v3 元数据；
- `inspect_bottom_telemetry.py`：采集前检查底盘反馈频率和速度方向；
- `view_cameras.py`：通过 SSH 隧道查看 Top 与腕部相机；
- `replay_episode.py`：检查或在真机复现一条原始示教的9维动作；默认dry-run；
- `lift_home.py`：升降台最高点回零；`python -m scripts.lift_home --execute`。
- `gripper_calibrate.py`：夹爪状态查看与标定；`python -m scripts.gripper_calibrate --status`。
- `indicator_light.py`：实体灯测试；`python -m scripts.indicator_light --execute --enable-robot`。
- `diagnostics/lift_diagnostic.py`：升降台多高度验收。
- `diagnostics/chassis_motion_test.py`：底盘平台通信、心跳与运动诊断。
- `diagnostics/gripper_diagnostic.py`：旧限力诊断已停用，仅保留参考代码。

直接转换：

```bash
/home/nvidia/miniforge3/envs/pi05-compat/bin/python -m scripts.convert_to_lerobot_v3
```

检查最新一条轨迹但不连接硬件：

```bash
/home/nvidia/miniforge3/envs/pi05-compat/bin/python -m scripts.replay_episode latest
```

真机复现最新轨迹（需要输入 `REPLAY`，底盘切模式时还需按实体确认键）：

```bash
/home/nvidia/miniforge3/envs/pi05-compat/bin/python -m scripts.replay_episode latest --execute
```

请在工程根目录使用 `python -m scripts.模块名` 运行；这些工具不是自动测试。

遥操和数采状态现由实体信号灯显示，控制逻辑位于 `Dobot_Init/indicator_light.py`；
DO6为红色、DO7为绿色、DO8为黄色。独立实机测试位于 `scripts/indicator_light.py`。

转换仅接受原始 `schema_version=2`、9维状态和9维动作，夹爪0开1闭。
转换和训练使用工程内 `Dobot_Train/lerobot-main/src`；默认CPU H.264编码MP4。
视频回归测试：`python -m unittest scripts.tests.pipeline.test_pipeline`，包含MP4编码和LeRobot回读。

### 快速转换

```bash
python -m scripts.convert_to_lerobot_v3
```

直接逐帧读取并转换，不增加全量预检、时间审计、报告生成或转换后回读。
保持原有状态/动作定义与帧顺序。默认逐条episode保存、双相机并行编码，图像写入线程4，
CPU H.264 preset为fast、CRF为30。当前本地LeRobot的跨episode批量路径不兼容写入模式；
`--video-batch-size`大于1时自动回退到1。

```bash
python -m scripts.convert_to_lerobot_v3 --video-preset fast
```

路径默认读取config/，也可用`--raw-root`、`--output-root`、`--repo-id`覆盖。
直接写入输出目录，已有目录不覆盖；失败后需人工移走未完成输出或更换路径重跑。

### 流式编码提速

转换默认使用有界队列把RGB直接送给双相机编码线程，省去中间PNG写入、再次读取
及其解码。每路默认2个编码线程、16帧队列；队列满时等待，不丢帧。
每5秒显示当前episode已送入帧数，保存完成后显示累计吞吐。
状态/动作统计仍由LeRobot生成；图像统计改由流式编码器在线累计，
与原来抽样图片计算的统计量可能略有差别。

```bash
python -u -m scripts.convert_to_lerobot_v3 --encoder-threads 2
# 恢复原来的中间图片编码方式
python -u -m scripts.convert_to_lerobot_v3 --no-streaming-encoding
```

`--image-writer-threads`只在中间图片模式使用。`--encoder-queue-maxsize`控制内存中的
排队帧数，增大队列不一定能提速。跨episode批量编码仍禁用。
