# quest_vr

本目录保存 Quest 侧 APK，以及 NVIDIA 侧的 UDP 协议解析、坐标映射、运动学与本地 IK。
正常操作不需要修改这些文件；重新标定坐标方向时才需要检查 `mapping.py` 和
`kinematics.py`。数值标定结果集中写在根目录 `constants.py`。

