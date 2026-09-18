# Dobot_Train

`train_pi05.py` 是 π0.5 训练入口，`check_training_setup.py` 只做 CUDA、数据集和配置
预检。训练路径和超参数均来自根目录 `constants.py`。
当前π0.5数据约定为9维状态、9维动作，升降台固定在0.107m并排除在策略之外。
转换器兼容旧12维状态/10维动作，并按 `RELABEL_TASK_DURING_CONVERSION` 将旧语言指令
统一重标为当前任务文字。

```bash
python -m Dobot_Train.check_training_setup
bash run_training.sh
```
