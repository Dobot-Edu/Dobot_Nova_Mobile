"""工程内 LeRobot 源码入口；子进程也继承同一来源。"""
import os
import sys
from pathlib import Path


def use_local_lerobot() -> Path:
    source = Path(__file__).resolve().parent / "lerobot-main" / "src"
    if not (source / "lerobot" / "__init__.py").is_file():
        raise RuntimeError(f"缺少工程内LeRobot源码：{source}")
    loaded = sys.modules.get("lerobot")
    if loaded is not None and not Path(loaded.__file__).resolve().is_relative_to(source):
        raise RuntimeError(f"已导入外部LeRobot：{loaded.__file__}，请重启Python进程")
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    paths = os.environ.get("PYTHONPATH", "").split(os.pathsep)
    os.environ["PYTHONPATH"] = os.pathsep.join([str(source), *[p for p in paths if p and p != str(source)]])
    return source
