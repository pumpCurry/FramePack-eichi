"""
ブラウザ再接続 共通コアモジュール

FanoutQueue (pub-sub with history replay) と JobContext (per-job state)
を提供する。oneframe / endframe / endframe_f1 で共有。

使い方:
    from eichi_utils.resync_core import FanoutQueue, JobContext, BUS_END_SENTINEL
"""

import queue
import threading
import uuid
import weakref
from collections import deque


# ====================================================================
# センチネル
# ====================================================================
BUS_END_SENTINEL = (None, None)


# ====================================================================
# FanoutQueue: 履歴再生付きスレッドセーフpub-sub
# ====================================================================
class FanoutQueue:
    """履歴を再生できるスレッドセーフなファンアウトキュー。

    publish() で全購読者に配信しつつ履歴に保存。
    subscribe() で既存の履歴を即座に受け取り、以降のリアルタイム配信を受ける。
    ブラウザの再接続時に履歴を再生することで、状態を復元する。
    """

    def __init__(self, maxlen: int = 200, maxsize: int = 64,
                 on_publish_tap=None):
        """
        Args:
            maxlen: 履歴バッファの最大長
            maxsize: 各購読キューの最大サイズ
            on_publish_tap: publish時に呼ばれるコールバック。
                            fn(item) → None。last_* グローバル更新等に使用。
        """
        self._history = deque(maxlen=maxlen)
        self._subs = weakref.WeakSet()
        self._lock = threading.Lock()
        self._maxsize = maxsize
        self._closed = False
        self._on_publish_tap = on_publish_tap

    def publish(self, item) -> None:
        """要素を全ての購読者に配信し履歴に保存する"""
        with self._lock:
            if self._closed:
                return
            self._history.append(item)

            # スナップショット・タップ更新
            if self._on_publish_tap is not None:
                try:
                    self._on_publish_tap(item)
                except Exception:
                    pass

            # 購読者へ配信（満杯なら古い1件を捨てて最新を入れる）
            for q in list(self._subs):
                try:
                    q.put_nowait(item)
                except queue.Full:
                    try:
                        _ = q.get_nowait()
                        q.put_nowait(item)
                    except Exception:
                        pass

    def subscribe(self) -> queue.Queue:
        """キューに購読し既存の履歴を即座に受け取る"""
        q: queue.Queue = queue.Queue(maxsize=self._maxsize)
        with self._lock:
            for item in list(self._history):
                try:
                    q.put_nowait(item)
                except queue.Full:
                    break
            self._subs.add(q)
            if self._closed:
                try:
                    q.put_nowait(BUS_END_SENTINEL)
                except queue.Full:
                    _ = q.get_nowait()
                    q.put_nowait(BUS_END_SENTINEL)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._lock:
            try:
                self._subs.discard(q)
            except Exception:
                pass

    def clear(self) -> None:
        """履歴と全ての購読キューをクリアする"""
        with self._lock:
            self._history.clear()
            for q in list(self._subs):
                while not q.empty():
                    try:
                        q.get_nowait()
                    except queue.Empty:
                        break

    def close(self) -> None:
        """全ての購読キューを閉じて削除する。センチネルを必ず送信する。"""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            for q in list(self._subs):
                try:
                    q.put_nowait(BUS_END_SENTINEL)
                except queue.Full:
                    try:
                        _ = q.get_nowait()
                        q.put_nowait(BUS_END_SENTINEL)
                    except Exception:
                        pass
            self._subs.clear()


# ====================================================================
# JobContext: per-job state container
# ====================================================================
class JobContext:
    """ファンアウトキューなどジョブ固有の状態を保持するクラス。

    各生成ジョブごとに1つ作成される。
    bus: FanoutQueue — 進捗/ファイル/完了イベントの配信
    stop_mode: None | "image" | "step" — 停止リクエスト
    owner_sid: str — 生成を開始したブラウザタブのセッションID
    """

    def __init__(self, on_publish_tap=None):
        self.bus = FanoutQueue(on_publish_tap=on_publish_tap)
        self.done = threading.Event()
        self.stop_mode = None
        self._stop_lock = threading.Lock()
        self.owner_sid = None
        self.owner_connected = False
        self.owner_disconnected_at = 0.0
        self._sent_end = False

    def should_stop_step(self) -> bool:
        """中断モードが "step" で、まだ 'end' を送っていない場合にTrue。"""
        with self._stop_lock:
            return self.stop_mode == "step" and not self._sent_end

    def reset_stop_mode(self) -> None:
        """ジョブ開始時にstop_modeと内部フラグをリセット。"""
        with self._stop_lock:
            self.stop_mode = None
            self._sent_end = False


# ====================================================================
# セッション管理
# ====================================================================
def alloc_ui_session_id() -> str:
    """ブラウザタブごとにユニークなセッションIDを割り当てる。
    Gradio の block.load() から呼ばれ、gr.State に格納される。"""
    sid = uuid.uuid4().hex
    print(f"onload: 新しい UI セッションID を払い出し: {sid}")
    return sid


# ====================================================================
# Resync ガード定数
# ====================================================================
RESYNC_MIN_INTERVAL_MS = 500
RESYNC_CTX_LINGER_SEC = 1.0
