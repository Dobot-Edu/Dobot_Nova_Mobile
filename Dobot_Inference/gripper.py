"""把策略输出的连续夹爪位置交给带编码器反馈和限流的实机驱动。"""

from __future__ import annotations

import queue
import threading
import time

from Dobot_Init.gripper import TriggerGripper


class PolicyGripper:
    """异步执行1=张开、0=闭合的连续目标位置。"""

    _STOP = object()

    def __init__(
        self,
        config,
        *,
        execute: bool,
    ) -> None:
        self.driver = TriggerGripper(config, execute=execute)
        self._queue: queue.Queue[object] = queue.Queue(maxsize=1)
        self._errors: queue.Queue[Exception] = queue.Queue(maxsize=1)
        self._thread: threading.Thread | None = None

    @property
    def position(self) -> float:
        self.driver.check_error()
        snapshot = self.driver.feedback_snapshot()
        timestamp_ns = int(snapshot["feedback_monotonic_ns"])
        if not snapshot["feedback_valid"] or timestamp_ns <= 0:
            raise RuntimeError("夹爪尚无编码器反馈")
        age_s = (time.monotonic_ns() - timestamp_ns) / 1e9
        if age_s > self.driver.config.feedback_abort_age_s:
            raise RuntimeError(f"夹爪编码器反馈中断{age_s:.3f}s")
        return float(snapshot["position"])

    def connect(self) -> None:
        self.driver.connect()  # 实机驱动会在这里明确发送全开命令。
        self._thread = threading.Thread(target=self._worker, name="pi05-gripper", daemon=True)
        self._thread.start()
        print("[夹爪] π0.5 控制就绪，初始化为完全张开")

    def submit(self, prediction: float) -> None:
        self.check_error()
        try:
            self._queue.put_nowait(float(prediction))
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            self._queue.put_nowait(float(prediction))

    def check_error(self) -> None:
        try:
            error = self._errors.get_nowait()
        except queue.Empty:
            return
        raise RuntimeError(f"夹爪控制失败：{error}") from error

    def close(self) -> None:
        if self._thread is not None:
            try:
                self._queue.put_nowait(self._STOP)
            except queue.Full:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    pass
                self._queue.put_nowait(self._STOP)
            self._thread.join(timeout=2.0)
            self._thread = None
        # 异常退出时不自动张开，以免把已经抓住的饮料摔落。
        self.driver.close()

    def _worker(self) -> None:
        while True:
            value = self._queue.get()
            if value is self._STOP:
                return
            try:
                self.driver.set_target_position(float(value))
            except Exception as error:
                try:
                    self._errors.put_nowait(error)
                except queue.Full:
                    pass
                return
