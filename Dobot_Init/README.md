# Dobot_Init

本目录只保存硬件初始化和底层控制：Nova Dashboard、标准位回位、夹爪、升降台、
底盘 RPC、安全约束和相机。所有可调值统一来自根目录 `config/`。

只执行机械臂报警检查、使能和标准位初始化：

```bash
python -m Dobot_Init.startup_pose
```

该命令会在运动前再次要求现场按 Enter 确认。

机械臂信号灯使用控制柜数字输出：DO6红、DO7绿、DO8黄。独立测试命令：

```bash
/home/nvidia/miniforge3/envs/pi05-compat/bin/python \
  -m tests.hardware.indicator_light --execute
```

依次输入 `1`、`2`、`3` 测试红、黄、绿，输入 `0` 熄灭，`q` 退出并自动熄灭。
如果控制柜要求机械臂处于使能状态，再额外添加 `--enable-robot`；该脚本不会发送运动指令。

夹爪舵机单独上报状态32时按正常夹持接触处理：立即锁定到最近的编码器实测位置，
不再继续加深闭合目标；Trigger松开开始张开时自动复位力矩。包含过流、过温、电压或
位置传感器状态位的组合错误仍会终止控制，不能作为普通接触忽略。

升降台位置寄存器显示0 cm、但物理上限 DI2 尚未到达时，普通绝对位置“上升”无法
继续。先停止遥操，再使用单次回零工具：

```bash
/home/nvidia/miniforge3/envs/pi05-compat/bin/python -m tests.hardware.lift_home --execute
```

现场输入 `HOME` 后才会向上运动，并在 DI2 最高限位触发后停止。不要通过允许负位置
绕过0 cm边界。

MAB-M006-2-A-0 前端网关实测会返回三字节 PDU `90 00 06`，其中两字节异常码
`0x0006` 表示从站忙。客户端会正确解析并以递增间隔有限重试；回零状态切换之间也加入
了等待。若四次重试后仍忙，程序安全退出，不会忽略异常继续运行。
