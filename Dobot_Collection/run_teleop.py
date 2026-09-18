#!/usr/bin/env python3
"""Quest 3 右手柄遥操 Nova 的命令行入口，同时持久化运行日志。"""

from __future__ import annotations

import logging
import logging.handlers
import os
import sys
import threading
import traceback
from pathlib import Path
from typing import TextIO

from Dobot_Collection.teleop import main as app_main


ROOT = Path(__file__).resolve().parents[1]
LOG_PATH = ROOT / "logs" / "run_vr_teleop.log"


class TeeStream:
    """保留终端实时输出，同时按行写入轮转日志。"""

    def __init__(self, console: TextIO, logger: logging.Logger, level: int) -> None:
        self.console = console
        self.logger = logger
        self.level = level
        self._buffer = ""
        self._lock = threading.Lock()

    @property
    def encoding(self) -> str:
        return self.console.encoding or "utf-8"

    def isatty(self) -> bool:
        return self.console.isatty()

    def fileno(self) -> int:
        return self.console.fileno()

    def write(self, text: str) -> int:
        if not text:
            return 0
        self.console.write(text)
        self.console.flush()
        with self._lock:
            self._buffer += text
            while "\n" in self._buffer:
                line, self._buffer = self._buffer.split("\n", 1)
                self.logger.log(self.level, line.rstrip("\r"))
        return len(text)

    def flush(self) -> None:
        self.console.flush()
        with self._lock:
            if self._buffer:
                self.logger.log(self.level, self._buffer.rstrip("\r"))
                self._buffer = ""


def _build_file_logger() -> tuple[logging.Logger, logging.Handler]:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("vr-teleop-session")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for old_handler in list(logger.handlers):
        logger.removeHandler(old_handler)
        old_handler.close()
    handler = logging.handlers.RotatingFileHandler(
        LOG_PATH,
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    return logger, handler


def main() -> int:
    logger, handler = _build_file_logger()
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    stdout_tee = TeeStream(original_stdout, logger, logging.INFO)
    stderr_tee = TeeStream(original_stderr, logger, logging.ERROR)
    sys.stdout = stdout_tee
    sys.stderr = stderr_tee
    try:
        print(
            f"[日志] PID={os.getpid()}，终端输出同步写入 {LOG_PATH}，"
            "单文件最大10MB，保留5份历史"
        )
        logger.info("启动参数: %r", sys.argv)
        return app_main()
    except BaseException:
        print("未捕获异常，完整堆栈如下：", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        raise
    finally:
        stdout_tee.flush()
        stderr_tee.flush()
        logger.info("进程退出")
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        handler.flush()
        handler.close()
        logger.removeHandler(handler)


if __name__ == "__main__":
    raise SystemExit(main())
