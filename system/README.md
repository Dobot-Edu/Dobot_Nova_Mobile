# 系统服务模板

Wi-Fi恢复和底盘平台心跳属于主机运维，不随遥操工具启动。
本次仅整理模板位置，没有重新安装、重启或修改正在运行的systemd服务。
现有底盘服务仍可能引用 `/home/nvidia/project/single_arm_VR/test_chassis_motion.py`。
部署新版模板前需单独核对服务路径，避免中断当前底盘连接。
