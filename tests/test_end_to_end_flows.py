# -*- coding: utf-8 -*-
import os, sys, re

sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))
sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/../webui"))
sys.argv = [sys.argv[0]]  # argparse対策

# 既存のスタブ群を取り込み（重い依存をダミー化）
import smoke_stream_test  # noqa: F401

_TS_RE = re.compile(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}")

def _desc_from_ui(ui_tuple):
    # (_filename, _preview, desc, bar, start_btn, end_btn, stop_cur, stop_step, seed_upd)
    return ui_tuple[2]

def _start_enabled(ui_tuple):
    # ("update", {"interactive": True/False, ...})
    return ui_tuple[4][1].get("interactive", None) in (True, None)


def test_normal_completion_via_stream_has_timestamp_and_enables_start(monkeypatch):
    """
    finalize 由来のprogressメッセージが流れてくる通常経路を、bus直叩きで検証。
    """
    import importlib
    from datetime import datetime as _dt

    one = importlib.import_module("webui.oneframe_ichi")
    # 件数（サマリ表示用）
    one.progress_ref_total = 1
    one.progress_img_total = 1
    one.progress_ref_idx = 1
    one.progress_img_idx = 1
    one.last_progress_desc = ""     # finalize未通過でも影響しないことを担保
    one.last_preview_image = None
    monkeypatch.setattr(one, "_compute_stop_controls", lambda running: (False, False, False, "", ""))

    ctx = one.JobContext()
    ts = _dt.now().strftime("%Y-%m-%d %H:%M:%S")
    summary = f"参考画像 {one.progress_ref_idx}/{one.progress_ref_total} ,イメージ {one.progress_img_idx}/{one.progress_img_total}"
    msg = one.translate("【全バッチ処理完了】プロセスが完了しました - ") + ts + " - " + summary

    # 完了メッセージ → end → close の順に投入
    ctx.bus.publish(('progress', (None, msg, '')))
    ctx.bus.publish(('end', None))
    ctx.bus.close()

    outs = list(one._stream_job_to_ui(ctx))
    assert outs, "UIストリームが空です"
    last = outs[-1]
    desc = _desc_from_ui(last) or ""
    assert ("完了" in desc) or ("中断" in desc), f"完了/中断メッセージが不在: {desc!r}"
    assert _TS_RE.search(desc), "完了メッセージに時刻がありません"
    assert _start_enabled(last), "完了後に Start が再有効化されていません"


def test_end_immediate_path_has_timestamp(monkeypatch):
    """
    END_IMMEDIATE（即時停止）でも、最終UIタプルに時刻入り中断サマリ＋Start再有効化が出る。
    """
    import importlib

    one = importlib.import_module("webui.oneframe_ichi")
    one.progress_ref_total = 1
    one.progress_img_total = 1
    one.progress_ref_idx = 0
    one.progress_img_idx = 0
    one.last_progress_desc = ""
    one.last_preview_image = None
    monkeypatch.setattr(one, "_compute_stop_controls", lambda running: (False, False, False, "", ""))

    one.stop_state.request(one.StopMode.END_IMMEDIATE)
    ctx = one.JobContext()
    ctx.bus.publish((None, None))  # センチネルのみ

    outs = list(one._stream_job_to_ui(ctx))
    assert outs, "UIストリームが空です"
    last = outs[-1]
    desc = _desc_from_ui(last) or ""
    assert "中断" in desc, f"中断メッセージが不在: {desc!r}"
    assert _TS_RE.search(desc), "中断メッセージに時刻がありません"
    assert _start_enabled(last), "中断後に Start が再有効化されていません"


def test_end_via_sentinel_direct_fallback(monkeypatch):
    """
    finalize が progress を流さずとも、(None, None) センチネルだけで
    `_stream_job_to_ui` が時刻＋件数入りの完了/中断サマリをフォールバック合成すること。
    """
    import importlib

    one = importlib.import_module("webui.oneframe_ichi")
    one.progress_ref_total = 2
    one.progress_img_total = 3
    one.progress_ref_idx = 2
    one.progress_img_idx = 3
    one.last_progress_desc = ""      # フォールバック合成を強制
    one.last_preview_image = None
    monkeypatch.setattr(one, "_compute_stop_controls", lambda running: (False, False, False, "", ""))

    ctx = one.JobContext()
    ctx.bus.publish((None, None))  # 完全に finalize 不在ケース
    outs = list(one._stream_job_to_ui(ctx))
    assert outs, "UIストリームが空です"
    last = outs[-1]
    desc = _desc_from_ui(last) or ""
    # 完了 or 中断のどちらかで、時刻と件数が入っていること
    assert ("完了" in desc) or ("中断" in desc), f"完了/中断メッセージが不在: {desc!r}"
    assert "参考画像 2/2" in desc and "イメージ 3/3" in desc, f"件数サマリが不正: {desc!r}"
    assert _TS_RE.search(desc), "時刻がありません"
    assert _start_enabled(last), "完了後に Start が再有効化されていません"
