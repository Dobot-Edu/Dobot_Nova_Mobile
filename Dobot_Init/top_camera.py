"""Top 相机采集、JPEG 服务和算力板本地预览。"""

from __future__ import annotations

import argparse
import asyncio
import json
import signal
import socket
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

import constants


@dataclass(frozen=True)
class CameraConfig:
    serial: str
    width: int
    height: int
    fps: int
    jpeg_quality: int
    listen_host: str
    http_port: int
    preview: bool
    window_name: str


class LatestFrame:
    """在线程之间保存最新 JPEG；新请求永远不会排队读取旧帧。"""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._jpeg = b""
        self._sequence = 0
        self._captured_at = 0.0

    def publish(self, jpeg: bytes) -> int:
        if not jpeg:
            raise ValueError("JPEG 数据不能为空")
        with self._condition:
            self._jpeg = jpeg
            self._sequence += 1
            self._captured_at = time.monotonic()
            self._condition.notify_all()
            return self._sequence

    def snapshot(self) -> tuple[bytes, int, float]:
        with self._condition:
            return self._jpeg, self._sequence, self._captured_at


class ViewerStats:
    """记录最近的视频请求，用于确认 Quest 是否真正收到画面。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._request_count = 0
        self._last_client = ""
        self._last_request_at = 0.0

    def record(self, client: str | None) -> None:
        with self._lock:
            self._request_count += 1
            self._last_client = client or ""
            self._last_request_at = time.monotonic()

    def snapshot(self) -> tuple[int, str, float | None]:
        with self._lock:
            age_ms = None
            if self._last_request_at:
                age_ms = (time.monotonic() - self._last_request_at) * 1000
            return self._request_count, self._last_client, age_ms


def load_camera_config() -> CameraConfig:
    """从根目录 ``constants.py`` 读取 Top 相机配置。"""
    result = CameraConfig(
        serial=constants.TOP_CAMERA_SERIAL,
        width=constants.CAMERA_WIDTH,
        height=constants.CAMERA_HEIGHT,
        fps=constants.CAMERA_STREAM_FPS,
        jpeg_quality=constants.JPEG_QUALITY,
        listen_host=constants.TOP_CAMERA_LISTEN_HOST,
        http_port=constants.TOP_CAMERA_HTTP_PORT,
        preview=constants.TOP_CAMERA_PREVIEW,
        window_name=constants.TOP_CAMERA_WINDOW_NAME,
    )
    if not result.serial:
        raise ValueError("top_serial 不能为空")
    if result.width <= 0 or result.height <= 0 or result.fps <= 0:
        raise ValueError("相机分辨率和帧率必须为正数")
    if not 1 <= result.jpeg_quality <= 100:
        raise ValueError("jpeg_quality 必须在 1 到 100 之间")
    if not 1 <= result.http_port <= 65535:
        raise ValueError("http_port 超出范围")
    return result


def existing_service_is_ready(config: CameraConfig) -> bool:
    """同一个 top 服务已运行时直接复用，避免重复占用 8080。"""
    health_url = f"http://127.0.0.1:{config.http_port}/health"
    for _attempt in range(5):
        try:
            with urllib.request.urlopen(health_url, timeout=0.5) as response:
                payload = json.loads(response.read().decode("utf-8"))
            if payload.get("ok") is True:
                return True
        except (urllib.error.URLError, TimeoutError, ValueError):
            pass
        time.sleep(0.1)

    # 端口能建立 TCP 连接，但不是本工程的健康 top 服务，不能静默复用。
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(0.5)
        if probe.connect_ex(("127.0.0.1", config.http_port)) == 0:
            raise RuntimeError(
                f"端口 {config.http_port} 已被其他或异常服务占用；"
                "请先用 ss -ltnp 查看占用进程"
            )
    return False


def _run_http_server(
    config: CameraConfig,
    latest: LatestFrame,
    stop_event: threading.Event,
    ready_event: threading.Event,
) -> None:
    from aiohttp import web

    viewers = ViewerStats()

    async def health(_request: web.Request) -> web.Response:
        jpeg, sequence, captured_at = latest.snapshot()
        age_ms = (time.monotonic() - captured_at) * 1000 if jpeg else None
        request_count, last_client, last_request_age_ms = viewers.snapshot()
        return web.json_response(
            {
                "ok": bool(jpeg),
                "sequence": sequence,
                "age_ms": age_ms,
                "viewer_request_count": request_count,
                "last_viewer_ip": last_client,
                "last_viewer_request_age_ms": last_request_age_ms,
            }
        )

    async def frame(_request: web.Request) -> web.Response:
        jpeg, sequence, _captured_at = latest.snapshot()
        if not jpeg:
            raise web.HTTPServiceUnavailable(text="相机尚未产生画面")
        viewers.record(_request.remote)
        return web.Response(
            body=jpeg,
            content_type="image/jpeg",
            headers={
                "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
                "X-Frame-Sequence": str(sequence),
            },
        )

    async def index(_request: web.Request) -> web.Response:
        return web.Response(
            text=(
                "<!doctype html><meta charset='utf-8'><title>Top Camera</title>"
                "<style>body{margin:0;background:#111;display:grid;place-items:center;height:100vh}"
                "img{max-width:100vw;max-height:100vh}</style>"
                "<img id='v'><script>const v=document.querySelector('#v');"
                "function n(){v.src='/frame.jpg?t='+Date.now()}v.onload=()=>setTimeout(n,20);"
                "v.onerror=()=>setTimeout(n,500);n()</script>"
            ),
            content_type="text/html",
        )

    async def serve() -> None:
        app = web.Application()
        app.router.add_get("/", index)
        app.router.add_get("/health", health)
        app.router.add_get("/frame.jpg", frame)
        runner = web.AppRunner(app, access_log=None)
        await runner.setup()
        site = web.TCPSite(runner, config.listen_host, config.http_port)
        await site.start()
        ready_event.set()
        try:
            while not stop_event.is_set():
                await asyncio.sleep(0.1)
        finally:
            await runner.cleanup()

    asyncio.run(serve())


def run(config: CameraConfig, *, force_no_preview: bool = False) -> int:
    # 延迟导入，让无相机的开发机仍可导入配置并运行单元测试。
    import cv2
    import numpy as np
    import pyrealsense2 as rs

    latest = LatestFrame()
    stop_event = threading.Event()
    server_ready = threading.Event()
    server = threading.Thread(
        target=_run_http_server,
        args=(config, latest, stop_event, server_ready),
        name="top-camera-http",
        daemon=True,
    )
    server.start()
    if not server_ready.wait(timeout=5.0):
        raise RuntimeError("HTTP 视频服务启动超时")

    pipeline = rs.pipeline()
    rs_config = rs.config()
    rs_config.enable_device(config.serial)
    rs_config.enable_stream(
        rs.stream.color,
        config.width,
        config.height,
        rs.format.bgr8,
        config.fps,
    )
    preview = config.preview and not force_no_preview

    def request_stop(_signum: int, _frame: object) -> None:
        stop_event.set()

    old_sigint = signal.signal(signal.SIGINT, request_stop)
    old_sigterm = signal.signal(signal.SIGTERM, request_stop)
    try:
        pipeline.start(rs_config)
        print(
            f"top 相机已启动：{config.serial}，"
            f"{config.width}x{config.height}@{config.fps}"
        )
        print(f"Quest 视频地址：http://10.2.150.205:{config.http_port}/frame.jpg")
        if preview:
            print("算力板预览窗口已开启；按 Q 或 Esc 可退出。")
        while not stop_event.is_set():
            frames = pipeline.wait_for_frames(timeout_ms=2000)
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            image = np.asanyarray(color_frame.get_data())
            ok, encoded = cv2.imencode(
                ".jpg",
                image,
                [cv2.IMWRITE_JPEG_QUALITY, config.jpeg_quality],
            )
            if ok:
                latest.publish(encoded.tobytes())
            if preview:
                cv2.imshow(config.window_name, image)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    stop_event.set()
    finally:
        signal.signal(signal.SIGINT, old_sigint)
        signal.signal(signal.SIGTERM, old_sigterm)
        stop_event.set()
        pipeline.stop()
        if preview:
            cv2.destroyAllWindows()
        server.join(timeout=2.0)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="显示并发布 top 相机画面")
    parser.add_argument("--no-preview", action="store_true", help="不打开算力板窗口")
    args = parser.parse_args(argv)
    config = load_camera_config()
    if existing_service_is_ready(config):
        print(
            f"top 相机服务已经运行，直接复用："
            f"http://127.0.0.1:{config.http_port}/frame.jpg"
        )
        return 0
    return run(config, force_no_preview=args.no_preview)


if __name__ == "__main__":
    raise SystemExit(main())
