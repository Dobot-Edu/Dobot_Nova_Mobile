"""遥操拥有采集子进程；Pipe 确认状态，保存期间不排队开录。"""
import multiprocessing
from collections.abc import Callable


class CollectionSession:
    def __init__(self, state_callback: Callable[[str, str], None] | None = None):
        self.process = None
        self.pipe = None
        self.state = 'disabled'
        self.state_callback = state_callback

    def _transition(self, state: str, detail: str) -> None:
        self.state = state
        if self.state_callback is not None:
            self.state_callback(state, detail)

    def start(self):
        from Dobot_Collection.collect import run_session
        context = multiprocessing.get_context('spawn')
        self.pipe, child = context.Pipe()
        self.process = context.Process(target=run_session, args=(child,), name='nova-collection')
        self._transition('starting', '数采设备正在启动')
        self.process.start()
        child.close()

    def poll(self):
        if self.pipe is None or self.state == 'error':
            return
        try:
            while self.pipe.poll():
                state, detail = self.pipe.recv()
                self._transition(state, detail)
                print(f'[数采 {self.state}] {detail}')
            if self.process is not None and not self.process.is_alive():
                detail = '数采子进程已退出；遥操仍可使用，请退出并重启后继续采集'
                self._transition('error', detail)
                print(f'[数采异常] {detail}')
        except (EOFError, OSError):
            detail = '数采子进程连接已关闭；本条若未保存则保留为 .partial'
            self._transition('error', detail)
            print(f'[数采异常] {detail}')

    def toggle(self):
        self.poll()
        self.request({'ready': 'start', 'recording': 'stop'}.get(self.state))

    def request(self, command):
        expected = {'ready': 'start', 'recording': 'stop'}.get(self.state)
        if command is None or command != expected:
            print(f'[数采] 当前状态 {self.state}，忽略录制请求；等待 ready/recording 提示')
            return
        try:
            self.pipe.send(command)
            state = 'starting' if command == 'start' else 'saving'
            detail = '正在开始采集' if command == 'start' else '正在停止并保存数据'
            self._transition(state, detail)
        except (EOFError, OSError):
            self._transition('error', '无法向数采子进程发送指令')
            print('[数采异常] 无法发送采集指令')

    def close(self):
        if self.process is not None:
            try:
                self.pipe.send('shutdown')
            except (EOFError, OSError):
                pass
            self.process.join(timeout=4)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=2)
            self.pipe.close()
            self.process = None
