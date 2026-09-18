"""通过仅限本机的 HTTP 服务显示 top 与腕部 RealSense 画面。"""

from __future__ import annotations

import json
import signal
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import constants


@dataclass(frozen=True)
class DualCameraConfig:
    top_serial: str
    wrist_serial: str
    width: int
    height: int
    fps: int
    jpeg_quality: int
    top_http_port: int
    viewer_host: str
    viewer_port: int


def load_dual_camera_config() -> DualCameraConfig:
    """从根目录 ``constants.py`` 读取双相机查看器配置。"""
    result = DualCameraConfig(
        top_serial=constants.TOP_CAMERA_SERIAL,
        wrist_serial=constants.WRIST_CAMERA_SERIAL,
        width=constants.CAMERA_WIDTH,
        height=constants.CAMERA_HEIGHT,
        fps=constants.CAMERA_STREAM_FPS,
        jpeg_quality=constants.JPEG_QUALITY,
        top_http_port=constants.TOP_CAMERA_HTTP_PORT,
        viewer_host=constants.DUAL_VIEWER_HOST,
        viewer_port=constants.DUAL_VIEWER_PORT,
    )
    if not result.top_serial or not result.wrist_serial:
        raise ValueError("top_serial 和 wrist_serial 均不能为空")
    if result.top_serial == result.wrist_serial:
        raise ValueError("top 与腕部相机序列号不能相同")
    if result.width <= 0 or result.height <= 0 or result.fps <= 0:
        raise ValueError("相机分辨率和帧率必须为正数")
    if not 1 <= result.jpeg_quality <= 100:
        raise ValueError("jpeg_quality 必须在 1..100")
    if not 1 <= result.top_http_port <= 65535 or not 1 <= result.viewer_port <= 65535:
        raise ValueError("相机 HTTP 端口超出范围")
    # 查看器只允许通过 NVIDIA 本机或 SSH 端口转发访问，避免把画面暴露到办公网。
    if result.viewer_host not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("dual_viewer_host 必须是回环地址，请通过 SSH 隧道访问")
    return result


class LatestJpeg:
    """线程安全地保存某一路相机的最新帧和状态。"""

    def __init__(self, name: str) -> None:
        self.name = name
        self._lock = threading.Lock()
        self._jpeg = b""
        self._sequence = 0
        self._captured_at = 0.0
        self._error = ""

    def publish(self, jpeg: bytes) -> None:
        if not jpeg:
            raise ValueError("JPEG 数据不能为空")
        with self._lock:
            self._jpeg = bytes(jpeg)
            self._sequence += 1
            self._captured_at = time.monotonic()
            self._error = ""

    def set_error(self, error: object) -> None:
        with self._lock:
            self._error = str(error)

    def snapshot(self) -> tuple[bytes, int, float, str]:
        with self._lock:
            return self._jpeg, self._sequence, self._captured_at, self._error

    def status(self) -> dict[str, object]:
        jpeg, sequence, captured_at, error = self.snapshot()
        return {
            "ok": bool(jpeg) and not error,
            "sequence": sequence,
            "age_ms": (time.monotonic() - captured_at) * 1000 if jpeg else None,
            "error": error or None,
        }


def top_service_is_ready(port: int) -> bool:
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/health", timeout=0.7
        ) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return payload.get("ok") is True
    except (urllib.error.URLError, TimeoutError, ValueError, json.JSONDecodeError):
        return False


def proxy_jpeg_source(
    url: str,
    latest: LatestJpeg,
    stop_event: threading.Event,
) -> None:
    """复用已运行的 top 服务，避免第二次打开同一台 RealSense。"""
    while not stop_event.is_set():
        try:
            with urllib.request.urlopen(url, timeout=1.0) as response:
                latest.publish(response.read())
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            latest.set_error(f"top 服务读取失败：{error}")
            stop_event.wait(0.25)
        else:
            stop_event.wait(0.015)


def realsense_jpeg_source(
    serial: str,
    config: DualCameraConfig,
    latest: LatestJpeg,
    stop_event: threading.Event,
) -> None:
    """独占一台 RealSense，只采集彩色流并保存最新 JPEG。"""
    pipeline = None
    started = False
    try:
        import cv2
        import numpy as np
        import pyrealsense2 as rs

        pipeline = rs.pipeline()
        rs_config = rs.config()
        rs_config.enable_device(serial)
        rs_config.enable_stream(
            rs.stream.color,
            config.width,
            config.height,
            rs.format.bgr8,
            config.fps,
        )
        pipeline.start(rs_config)
        started = True
        while not stop_event.is_set():
            frames = pipeline.wait_for_frames(timeout_ms=2000)
            color = frames.get_color_frame()
            if not color:
                continue
            image = np.asanyarray(color.get_data())
            ok, encoded = cv2.imencode(
                ".jpg",
                image,
                [cv2.IMWRITE_JPEG_QUALITY, config.jpeg_quality],
            )
            if ok:
                latest.publish(encoded.tobytes())
    except Exception as error:
        latest.set_error(f"{latest.name} 相机启动/采集失败：{error}")
    finally:
        if started and pipeline is not None:
            pipeline.stop()


VIEWER_HTML = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Nova 双相机视角</title>
<style>
body{margin:0;background:#101216;color:#eee;font-family:sans-serif}
header{padding:12px 18px;display:flex;justify-content:space-between;align-items:center}
#status{color:#9ca3af;font-size:14px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:12px;padding:0 12px 12px}
.panel{background:#181b21;border-radius:10px;overflow:hidden}.title{padding:9px 12px;font-weight:600}
.frame{aspect-ratio:4/3;background:#050505;display:grid;place-items:center}.frame img{width:100%;height:100%;object-fit:contain}
@media(max-width:900px){.grid{grid-template-columns:1fr}}
</style></head><body>
<header><strong>Nova 双相机实时视角</strong><span id="status">正在连接…</span></header>
<main class="grid">
<section class="panel"><div class="title">Top 相机</div><div class="frame"><img id="top"></div></section>
<section class="panel"><div class="title">腕部相机</div><div class="frame"><img id="wrist"></div></section>
</main>
<script>
function loop(id,name){const img=document.getElementById(id);const next=()=>img.src='/frame/'+name+'.jpg?t='+Date.now();img.onload=()=>setTimeout(next,20);img.onerror=()=>setTimeout(next,300);next()}
loop('top','top');loop('wrist','wrist');
async function health(){try{const r=await fetch('/health',{cache:'no-store'}),j=await r.json();const f=x=>x.ok?`${Math.round(x.age_ms)} ms`:(x.error||'等待画面');document.getElementById('status').textContent=`Top ${f(j.top)} · 腕部 ${f(j.wrist)}`}catch(e){document.getElementById('status').textContent='状态连接失败'}setTimeout(health,1000)}health();
</script></body></html>"""


def make_handler(sources: dict[str, LatestJpeg]) -> type[BaseHTTPRequestHandler]:
    class ViewerHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            path = self.path.split("?", 1)[0]
            if path == "/":
                self._send(VIEWER_HTML.encode("utf-8"), "text/html; charset=utf-8")
                return
            if path == "/health":
                payload = json.dumps(
                    {name: source.status() for name, source in sources.items()},
                    ensure_ascii=False,
                ).encode("utf-8")
                self._send(payload, "application/json; charset=utf-8")
                return
            if path.startswith("/frame/") and path.endswith(".jpg"):
                name = path.removeprefix("/frame/").removesuffix(".jpg")
                source = sources.get(name)
                if source is None:
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                jpeg, sequence, _captured_at, error = source.snapshot()
                if not jpeg:
                    self.send_error(HTTPStatus.SERVICE_UNAVAILABLE, error or "等待相机画面")
                    return
                self._send(jpeg, "image/jpeg", {"X-Frame-Sequence": str(sequence)})
                return
            self.send_error(HTTPStatus.NOT_FOUND)

        def _send(
            self,
            body: bytes,
            content_type: str,
            extra_headers: dict[str, str] | None = None,
        ) -> None:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
            for name, value in (extra_headers or {}).items():
                self.send_header(name, value)
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, _format: str, *_args: object) -> None:
            return

    return ViewerHandler


def wait_for_first_frames(
    sources: dict[str, LatestJpeg],
    stop_event: threading.Event,
    timeout_s: float = 10.0,
) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline and not stop_event.is_set():
        snapshots = {name: source.snapshot() for name, source in sources.items()}
        if all(jpeg for jpeg, _seq, _at, _error in snapshots.values()):
            return
        time.sleep(0.05)
    errors = [error for _jpeg, _seq, _at, error in snapshots.values() if error]
    detail = "；".join(errors) if errors else "请检查相机是否被采集程序占用"
    raise RuntimeError(f"等待双相机首帧超时：{detail}")


def run(config: DualCameraConfig) -> int:
    stop_event = threading.Event()
    sources = {"top": LatestJpeg("top"), "wrist": LatestJpeg("腕部")}
    top_proxy = top_service_is_ready(config.top_http_port)
    if top_proxy:
        top_target = proxy_jpeg_source
        top_args = (
            f"http://127.0.0.1:{config.top_http_port}/frame.jpg",
            sources["top"],
            stop_event,
        )
        print(f"复用现有 top 服务：http://127.0.0.1:{config.top_http_port}")
    else:
        top_target = realsense_jpeg_source
        top_args = (config.top_serial, config, sources["top"], stop_event)
        print(f"现有 top 服务未运行，直接打开 top 相机：{config.top_serial}")

    workers = [
        threading.Thread(target=top_target, args=top_args, name="top-source", daemon=True),
        threading.Thread(
            target=realsense_jpeg_source,
            args=(config.wrist_serial, config, sources["wrist"], stop_event),
            name="wrist-source",
            daemon=True,
        ),
    ]
    for worker in workers:
        worker.start()

    server = None
    server_thread = None
    old_handlers: dict[int, object] = {}

    def request_stop(_signum: int, _frame: object) -> None:
        stop_event.set()

    try:
        wait_for_first_frames(sources, stop_event)
        server = ThreadingHTTPServer(
            (config.viewer_host, config.viewer_port), make_handler(sources)
        )
        server.daemon_threads = True
        server_thread = threading.Thread(
            target=server.serve_forever, name="dual-camera-http", daemon=True
        )
        server_thread.start()
        for signum in (signal.SIGINT, signal.SIGTERM):
            old_handlers[signum] = signal.signal(signum, request_stop)
        print(f"腕部相机已启动：{config.wrist_serial}")
        print(f"双相机查看器：http://{config.viewer_host}:{config.viewer_port}")
        print("请在本地浏览器打开 http://127.0.0.1:8765；按 Ctrl+C 退出。")
        while not stop_event.wait(0.2):
            pass
    finally:
        stop_event.set()
        if server is not None:
            server.shutdown()
            server.server_close()
        if server_thread is not None:
            server_thread.join(timeout=2.0)
        for worker in workers:
            worker.join(timeout=3.0)
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)
    return 0
