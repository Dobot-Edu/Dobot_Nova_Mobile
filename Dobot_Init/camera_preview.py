"""共享已有相机帧；独立Tk进程显示，不占用第二个相机连接。"""
import multiprocessing as mp
import os
from pathlib import Path
import queue
import threading
import time
import constants


def desktop_environment():
    # SSH启动时寻找同一用户已登录的本机桌面，仅复制显示所需变量。
    if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
        return
    for proc in Path('/proc').iterdir():
        try:
            if not proc.name.isdigit() or proc.stat().st_uid != os.getuid():
                continue
            if proc.joinpath('comm').read_text().strip() not in ('gnome-shell', 'plasmashell', 'xfce4-session'):
                continue
            env = dict(x.split('=', 1) for x in proc.joinpath('environ').read_bytes().decode().split('\0') if '=' in x)
            for key in ('DISPLAY', 'XAUTHORITY', 'WAYLAND_DISPLAY', 'XDG_RUNTIME_DIR'):
                if key in env:
                    os.environ[key] = env[key]
            return
        except (OSError, UnicodeError):
            continue


def _window(frames, title):
    try:
        desktop_environment()
        import tkinter as tk
        import cv2
        import numpy as np
        root = tk.Tk()
        root.title(title + ' — Top | Wrist')
        root.geometry('1280x520')
        label = tk.Label(root, text='等待相机画面……', bg='black', fg='white')
        label.pack(fill='both', expand=True)
        status = tk.Label(root, text='关闭窗口仅关闭预览；遥操/推理继续运行')
        status.pack()
        latest = None

        def refresh():
            nonlocal latest
            started = time.monotonic()
            try:
                while True:
                    latest = frames.get_nowait()
            except queue.Empty:
                pass
            if latest is not None:
                top, wrist, ages, received = latest
                canvas = np.hstack((top, wrist))
                scale = min(max(label.winfo_width(), 2)/canvas.shape[1], max(label.winfo_height(), 2)/canvas.shape[0])
                canvas = cv2.resize(canvas, (max(1,int(canvas.shape[1]*scale)),max(1,int(canvas.shape[0]*scale))))
                ok, encoded = cv2.imencode('.png', canvas)
                if ok:
                    photo = tk.PhotoImage(data=encoded.tobytes())
                    label.configure(image=photo, text='')
                    label.image = photo
                delay = time.monotonic()-received
                status.configure(text=f'Top帧龄 {ages[0]+delay:.2f}s | Wrist帧龄 {ages[1]+delay:.2f}s | 关闭窗口不停止程序',
                                 fg='red' if max(ages)+delay > .5 else 'black')
            delay_ms = max(1, round((1 / constants.CAMERA_PREVIEW_FPS - (time.monotonic() - started)) * 1000))
            root.after(delay_ms, refresh)
        refresh()
        root.mainloop()
    except Exception as error:
        print(f'[相机预览不可用] {error}；主程序继续运行', flush=True)


class CameraPreview:
    def __init__(self, sources, title):
        self.sources = sources
        self.stop = threading.Event()
        self.thread = None
        self.process = None
        self.frames = None
        self.title = title

    def start(self):
        if not constants.CAMERA_PREVIEW_ENABLED:
            return
        try:
            context = mp.get_context('spawn')
            self.frames = context.Queue(maxsize=1)
            self.process = context.Process(target=_window, args=(self.frames,self.title), daemon=True)
            self.process.start()
            self.thread = threading.Thread(target=self._feed, daemon=True)
            self.thread.start()
        except Exception as error:
            print(f'[相机预览不可用] {error}')

    def _feed(self):
        import cv2
        while not self.stop.is_set() and self.process.is_alive():
            started = time.monotonic()
            items = [self.sources[k].latest.get() for k in ('top','wrist')]
            if any(x is None for x in items):
                self.stop.wait(1 / constants.CAMERA_PREVIEW_FPS)
                continue
            now = time.monotonic()
            try:
                self.frames.put_nowait((*(cv2.resize(x.value,(constants.CAMERA_WIDTH, constants.CAMERA_HEIGHT)) for x in items),
                                        [now-x.monotonic for x in items],now))
            except queue.Full:
                pass
            self.stop.wait(max(0, 1 / constants.CAMERA_PREVIEW_FPS - (time.monotonic() - started)))

    def close(self):
        self.stop.set()
        if self.thread is not None:
            self.thread.join(timeout=1)
        if self.process is not None and self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=1)
        if self.frames is not None:
            self.frames.cancel_join_thread()
            self.frames.close()
