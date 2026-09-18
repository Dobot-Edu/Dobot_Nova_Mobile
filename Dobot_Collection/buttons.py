"""A/B 手势路由；不访问硬件。短按在松开时确认，长按不会变成短按。"""
from dataclasses import dataclass


@dataclass
class ButtonEvents:
    activate: bool = False
    home: bool = False
    record: bool = False
    lift_up: bool = False
    lift_down: bool = False


class ButtonRouter:
    def __init__(self, maintenance=False, window_s=0.3):
        self.maintenance = maintenance
        self.window_s = window_s
        self.reset()

    def reset(self):
        self.armed = False
        self.a_at = self.b_at = self.pending_a = None
        self.b_allowed = False
        self.last_b = float('-inf')

    def update(self, a, b, *, valid, activated, now, record_allowed=True):
        out = ButtonEvents()
        if not valid or (a and b):
            self.reset()
            return out
        if not self.armed:
            if not a and not b:
                self.armed = True
            return out

        if self.maintenance and activated:
            self.pending_a = None
            self.a_at = (now if self.a_at is None else self.a_at) if a else None
            self.b_at = (now if self.b_at is None else self.b_at) if b else None
            out.lift_up = a and now - self.a_at >= self.window_s
            out.lift_down = b and now - self.b_at >= self.window_s
            return out

        if a and self.a_at is None:
            self.a_at = now
        elif not a and self.a_at is not None:
            short = now - self.a_at < self.window_s
            self.a_at = None
            if short:
                if self.pending_a is not None and now - self.pending_a <= self.window_s:
                    out.home = activated and not self.maintenance
                    self.pending_a = None
                else:
                    self.pending_a = now
            else:
                self.pending_a = None
        if self.pending_a is not None and now - self.pending_a > self.window_s and not a:
            out.activate = not activated
            self.pending_a = None

        if b and self.b_at is None:
            self.b_at = now
            self.b_allowed = record_allowed
        elif not b and self.b_at is not None:
            short = now - self.b_at < self.window_s
            self.b_at = None
            if short and self.b_allowed and record_allowed and now - self.last_b >= self.window_s:
                out.record = not self.maintenance
            # 连击窗口从每次松开计算，连续抖动不能排队触发。
            self.last_b = now
        return out
