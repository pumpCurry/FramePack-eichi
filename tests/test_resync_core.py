"""eichi_utils.resync_core の単体テスト"""

import os
import importlib.util
import threading
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
spec = importlib.util.spec_from_file_location(
    "resync_core",
    os.path.join(ROOT, "webui", "eichi_utils", "resync_core.py"),
)
rc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rc)

FanoutQueue = rc.FanoutQueue
JobContext = rc.JobContext
BUS_END_SENTINEL = rc.BUS_END_SENTINEL


class TestFanoutQueue:
    def test_publish_subscribe(self):
        fq = FanoutQueue()
        q = fq.subscribe()
        fq.publish(("progress", "test"))
        assert q.get_nowait() == ("progress", "test")

    def test_history_replay(self):
        fq = FanoutQueue()
        fq.publish(("a", 1))
        fq.publish(("b", 2))
        # Late subscriber gets history
        q = fq.subscribe()
        assert q.get_nowait() == ("a", 1)
        assert q.get_nowait() == ("b", 2)

    def test_close_sends_sentinel(self):
        fq = FanoutQueue()
        q = fq.subscribe()
        fq.close()
        assert q.get_nowait() == BUS_END_SENTINEL

    def test_publish_after_close_ignored(self):
        fq = FanoutQueue()
        q = fq.subscribe()
        fq.close()
        fq.publish(("should", "ignore"))
        # Only sentinel in queue
        items = []
        while not q.empty():
            items.append(q.get_nowait())
        assert items == [BUS_END_SENTINEL]

    def test_on_publish_tap(self):
        captured = []
        fq = FanoutQueue(on_publish_tap=lambda item: captured.append(item))
        fq.publish(("x", 1))
        fq.publish(("y", 2))
        assert len(captured) == 2
        assert captured[0] == ("x", 1)

    def test_unsubscribe(self):
        fq = FanoutQueue()
        q = fq.subscribe()
        fq.unsubscribe(q)
        fq.publish(("after", "unsub"))
        assert q.empty()

    def test_clear(self):
        fq = FanoutQueue()
        fq.publish(("a", 1))
        fq.clear()
        q = fq.subscribe()
        assert q.empty()

    def test_maxlen_history(self):
        fq = FanoutQueue(maxlen=3)
        for i in range(10):
            fq.publish(("item", i))
        q = fq.subscribe()
        items = []
        while not q.empty():
            items.append(q.get_nowait())
        assert len(items) == 3
        assert items[0] == ("item", 7)  # oldest in ring buffer

    def test_multi_subscriber(self):
        fq = FanoutQueue()
        q1 = fq.subscribe()
        q2 = fq.subscribe()
        fq.publish(("test", 1))
        assert q1.get_nowait() == ("test", 1)
        assert q2.get_nowait() == ("test", 1)


class TestJobContext:
    def test_default_state(self):
        ctx = JobContext()
        assert ctx.stop_mode is None
        assert ctx.should_stop_step() is False
        assert ctx.owner_sid is None

    def test_stop_step(self):
        ctx = JobContext()
        ctx.stop_mode = "step"
        assert ctx.should_stop_step() is True
        ctx._sent_end = True
        assert ctx.should_stop_step() is False

    def test_reset_stop_mode(self):
        ctx = JobContext()
        ctx.stop_mode = "image"
        ctx._sent_end = True
        ctx.reset_stop_mode()
        assert ctx.stop_mode is None
        assert ctx._sent_end is False

    def test_bus_is_fanout_queue(self):
        ctx = JobContext()
        assert isinstance(ctx.bus, FanoutQueue)

    def test_on_publish_tap_wired(self):
        captured = []
        ctx = JobContext(on_publish_tap=lambda item: captured.append(item))
        ctx.bus.publish(("test", 1))
        assert len(captured) == 1


class TestAllocSessionId:
    def test_returns_hex_string(self):
        sid = rc.alloc_ui_session_id()
        assert isinstance(sid, str)
        assert len(sid) == 32  # uuid4 hex

    def test_unique(self):
        sid1 = rc.alloc_ui_session_id()
        sid2 = rc.alloc_ui_session_id()
        assert sid1 != sid2


class TestSentinel:
    def test_sentinel_value(self):
        assert BUS_END_SENTINEL == (None, None)

    def test_sentinel_is_tuple(self):
        assert isinstance(BUS_END_SENTINEL, tuple)


class TestBugFixes:
    """監査で発見されたバグの修正を検証するテスト"""

    def test_bug15_uses_regular_set(self):
        """BUG-15: WeakSet → 通常の set でGC耐性"""
        fq = FanoutQueue()
        assert isinstance(fq._subs, set)

    def test_bug16_maxsize_matches_maxlen(self):
        """BUG-16: maxsize デフォルトが maxlen と一致（履歴ドロップ防止）"""
        fq = FanoutQueue(maxlen=200)
        assert fq._maxsize == 200

    def test_bug17_sentinel_on_full_queue(self):
        """BUG-17: キューが満杯でもsentinelが配信される"""
        fq = FanoutQueue(maxlen=5, maxsize=5)
        q = fq.subscribe()
        for i in range(5):
            fq.publish(("fill", i))
        fq.close()
        items = []
        while not q.empty():
            items.append(q.get_nowait())
        assert items[-1] == BUS_END_SENTINEL

    def test_bug5_get_timeout(self):
        """BUG-5: q.get(timeout=...) でタイムアウト可能"""
        import queue
        fq = FanoutQueue(maxlen=10, maxsize=10)
        q = fq.subscribe()
        # publish/closeなしでタイムアウトすること
        try:
            q.get(timeout=0.1)
            assert False, "Should have raised Empty"
        except queue.Empty:
            pass

    def test_bug5_sentinel_delivered_on_close(self):
        """BUG-5: close()後のget(timeout)でsentinelが取れる"""
        fq = FanoutQueue(maxlen=10, maxsize=10)
        q = fq.subscribe()
        fq.close()
        item = q.get(timeout=1)
        assert item == BUS_END_SENTINEL

    def test_subscribe_after_close_gets_sentinel(self):
        """close後のsubscribeでもsentinelが即時配信"""
        fq = FanoutQueue(maxlen=10, maxsize=10)
        fq.close()
        q = fq.subscribe()
        item = q.get(timeout=1)
        assert item == BUS_END_SENTINEL

    def test_close_idempotent(self):
        """close()の二重呼び出しでエラーにならない"""
        fq = FanoutQueue()
        fq.close()
        fq.close()
        assert fq.is_closed

    def test_is_closed_property(self):
        fq = FanoutQueue()
        assert not fq.is_closed
        fq.close()
        assert fq.is_closed

    def test_default_get_timeout_exported(self):
        """_DEFAULT_GET_TIMEOUT が定義されている"""
        assert hasattr(rc, '_DEFAULT_GET_TIMEOUT')
        assert rc._DEFAULT_GET_TIMEOUT > 0
