from Dobot_Collection.buttons import ButtonRouter
from Dobot_Collection.session import CollectionSession
from Dobot_Init.chassis import VRChassis
from types import SimpleNamespace


def feed(router, t, a=False, b=False, active=True, valid=True, allowed=True):
    return router.update(a, b, valid=valid, activated=active, now=t, record_allowed=allowed)


def test_single_a_activates_after_window_only_once():
    r = ButtonRouter()
    feed(r, 0, active=False)
    feed(r, 1, a=True, active=False)
    assert not feed(r, 1.1, active=False).activate
    assert feed(r, 1.41, active=False).activate
    assert not feed(r, 1.5).activate


def test_double_a_homes_without_record_event():
    r = ButtonRouter()
    feed(r, 0)
    feed(r, 1, a=True)
    feed(r, 1.05)
    feed(r, 1.15, a=True)
    e = feed(r, 1.2)
    assert e.home and not e.record and not e.activate
    assert not feed(r, 1.8).activate


def test_full_long_a_and_b_do_nothing():
    for key in ('a', 'b'):
        r = ButtonRouter()
        feed(r, 0, active=False)
        feed(r, 1, active=False, **{key: True})
        feed(r, 2, active=False, **{key: True})
        assert not any(vars(feed(r, 3, active=False)).values())
        assert not any(vars(feed(r, 4, active=False)).values())


def test_b_click_toggle_debounces_double_b():
    r = ButtonRouter()
    feed(r, 0)
    feed(r, 1, b=True)
    assert feed(r, 1.05).record
    feed(r, 1.15, b=True)
    assert not feed(r, 1.2).record
    feed(r, 2, b=True)
    assert feed(r, 2.05).record


def test_press_during_saving_cannot_start_after_ready():
    r = ButtonRouter()
    feed(r, 0)
    feed(r, 1, b=True, allowed=False)
    assert not feed(r, 1.1, allowed=True).record


def test_invalid_tracking_and_ab_require_release():
    for invalid in ({'valid': False}, {'a': True, 'b': True}):
        r = ButtonRouter()
        feed(r, 0)
        feed(r, 1, a=True)
        feed(r, 1.1, **invalid)
        assert not feed(r, 1.2, a=True).activate
        feed(r, 1.3)
        assert not any(vars(feed(r, 2)).values())


def test_maintenance_holds_and_ab_stop():
    r = ButtonRouter(maintenance=True)
    feed(r, 0)
    assert not feed(r, 1, a=True).lift_up
    assert feed(r, 1.31, a=True).lift_up
    assert not feed(r, 1.4, a=True, b=True).lift_up
    assert not feed(r, 1.8, b=True).lift_down
    feed(r, 2)
    feed(r, 3, b=True)
    assert feed(r, 3.31, b=True).lift_down
    assert not feed(r, 3.4).lift_down


def test_maintenance_waiting_long_a_does_not_initialize():
    r = ButtonRouter(maintenance=True)
    feed(r, 0, active=False)
    feed(r, 1, a=True, active=False)
    assert not feed(r, 2, a=True, active=False).lift_up
    assert not feed(r, 2.1, active=False).activate


def test_collection_pending_and_saving_do_not_queue_commands():
    s = CollectionSession()
    sent = []
    s.pipe = SimpleNamespace(send=sent.append)
    s.poll = lambda: None
    s.state = 'ready'
    s.toggle()
    s.toggle()
    assert sent == ['start']
    s.state = 'recording'
    s.toggle()
    s.toggle()
    assert sent == ['start', 'stop']
    s.state = 'ready'
    s.toggle()
    assert sent == ['start', 'stop', 'start']


def test_arm_only_does_not_command_chassis_on_enable_update_or_close():
    chassis = VRChassis(SimpleNamespace(enabled=False), execute=True, read_only=True)
    calls = []
    chassis.rpc = SimpleNamespace(logout=lambda: calls.append('logout'))
    chassis.enable_remote()
    chassis.update([1, 1], input_valid=True)
    assert chassis.desired.linear_mm_s == 0
    assert chassis.desired.angular_mdeg_s == 0
    chassis.close()
    assert calls == ['logout']


def test_b_stop_saves_without_terminal_confirmation(tmp_path, monkeypatch):
    import json
    import time
    import numpy as np
    from Dobot_Collection import collect
    monkeypatch.setattr(collect.settings, 'RAW_ROOT', tmp_path)
    monkeypatch.setattr(collect.settings, 'CAMERA_WIDTH', 4)
    monkeypatch.setattr(collect.settings, 'CAMERA_HEIGHT', 4)
    monkeypatch.setattr('builtins.input', lambda *a: (_ for _ in ()).throw(AssertionError('terminal input')))
    sources = {'feedback': {'actual_deg': [0] * 6}, 'top': np.zeros((4, 4, 3), dtype=np.uint8),
               'wrist': np.zeros((4, 4, 3), dtype=np.uint8), 'telemetry': {}}
    monkeypatch.setattr(collect, 'require_fresh', lambda source, *a: SimpleNamespace(value=source, monotonic=time.monotonic()))
    diagnostic = {'feedback_age_s': 0, 'feedback_monotonic_ns': 1}
    monkeypatch.setattr(collect, 'build_vectors', lambda *a, **k: (np.zeros(9), np.zeros(9), diagnostic, diagnostic))
    class Pipe:
        polls = 0
        statuses = []
        def poll(self):
            self.polls += 1
            return self.polls > 1
        def recv(self): return 'stop'
        def send(self, message): self.statuses.append(message[0])
    pipe = Pipe()
    path, frames, success = collect.record_episode(sources, pipe)
    assert success and frames == 1 and (path / 'COMPLETE').exists()
    assert json.loads((path / 'metadata.json').read_text())['end_reason'] == 'right_B_click'
    assert pipe.statuses == ['recording', 'saving']
    assert len(list(tmp_path.glob('episode_*'))) == 1


def test_shutdown_leaves_partial_without_complete(tmp_path, monkeypatch):
    import unittest
    from Dobot_Collection import collect
    monkeypatch.setattr(collect.settings, 'RAW_ROOT', tmp_path)
    pipe = SimpleNamespace(poll=lambda: True, recv=lambda: 'shutdown', send=lambda x: None)
    with unittest.TestCase().assertRaises(collect.SessionShutdown):
        collect.record_episode({}, pipe)
    partials = list(tmp_path.glob('.*.partial'))
    assert len(partials) == 1
    assert not (partials[0] / 'COMPLETE').exists()
