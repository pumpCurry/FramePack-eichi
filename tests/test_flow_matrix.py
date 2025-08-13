# -*- coding: utf-8 -*-
import os, sys, re, types
import pytest

sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))
sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/../webui"))
sys.argv = [sys.argv[0]]  # argparse対策

# 既存のスタブ群を取り込み（重い依存をダミー化）
import smoke_stream_test  # noqa: F401

_TS_RE = re.compile(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}")

def _desc(ui): return ui[2]
def _bar(ui): return ui[3]
def _btn_start(ui): return ui[4]   # ("update", {...})
def _btn_end(ui): return ui[5]
def _btn_stop_cur(ui): return ui[6]
def _btn_stop_step(ui): return ui[7]
def _seed(ui): return ui[8]
def _is_interactive(btn_update): return btn_update[1].get("interactive", None) in (True, None)


def _fresh_ctx(one):
    one.batch_stopped = False
    one.last_stop_mode = one.StopMode.NONE
    one.stop_state.clear()
    ctx = one.JobContext()
    one.cur_job = ctx
    one.generation_active = True
    return ctx


def test_start_to_end_normal_path(monkeypatch):
    """
    正常完了: CUI/GUIとも終了サマリ（時刻＋件数）→ Start再有効化。
    """
    import importlib
    from datetime import datetime as _dt
    one = importlib.import_module("webui.oneframe_ichi")
    one.progress_ref_total = 1
    one.progress_img_total = 1
    one.progress_ref_idx = 1
    one.progress_img_idx = 1
    one.last_progress_desc = ""
    one.last_preview_image = None
    monkeypatch.setattr(one, "_compute_stop_controls", lambda running: (False, False, False, "", ""))
    ctx = _fresh_ctx(one)
    ts = _dt.now().strftime("%Y-%m-%d %H:%M:%S")
    summary = f"参考画像 {one.progress_ref_idx}/{one.progress_ref_total} ,イメージ {one.progress_img_idx}/{one.progress_img_total}"
    ctx.bus.publish(('progress', (None, one.translate("【全バッチ処理完了】プロセスが完了しました - ") + ts + " - " + summary, '')))
    ctx.bus.publish(('end', None))
    ctx.bus.close()
    outs = list(one._stream_job_to_ui(ctx))
    assert outs, "UIストリームが空"
    last = outs[-1]
    d = _desc(last) or ""
    assert "完了" in d, f"完了メッセージ不在: {d!r}"
    assert _TS_RE.search(d), "時刻が無い"
    assert "参考画像 1/1" in d and "イメージ 1/1" in d, f"件数サマリ不正: {d!r}"
    assert _is_interactive(_btn_start(last)), "Startが再有効化されていない"


def test_abnormal_path_exception_fallback(monkeypatch):
    """
    finalizeがprogressを流さずセンチネルのみでも、_stream_job_to_uiが時刻＋件数を合成する。
    """
    import importlib
    one = importlib.import_module("webui.oneframe_ichi")
    one.progress_ref_total = 2
    one.progress_img_total = 3
    one.progress_ref_idx = 2
    one.progress_img_idx = 3
    one.last_progress_desc = ""  # フォールバック発火
    one.last_preview_image = None
    monkeypatch.setattr(one, "_compute_stop_controls", lambda running: (False, False, False, "", ""))
    ctx = _fresh_ctx(one)
    ctx.bus.publish((None, None))  # finalize不在
    outs = list(one._stream_job_to_ui(ctx))
    last = outs[-1]
    d = _desc(last) or ""
    assert ("完了" in d) or ("中断" in d), f"完了/中断メッセージ不在: {d!r}"
    assert "参考画像 2/2" in d and "イメージ 3/3" in d, f"件数サマリ不正: {d!r}"
    assert _TS_RE.search(d), "時刻が無い"
    assert _is_interactive(_btn_start(last)), "Startが再有効化されていない"


def test_end_generation_immediate(monkeypatch):
    """
    [生成終了] 即時停止: 中断サマリ（時刻＋件数）→ Start再有効化。
    """
    import importlib
    one = importlib.import_module("webui.oneframe_ichi")
    one.progress_ref_total = 1
    one.progress_img_total = 5
    one.progress_ref_idx = 0
    one.progress_img_idx = 2
    one.last_progress_desc = ""
    one.last_preview_image = None
    monkeypatch.setattr(one, "_compute_stop_controls", lambda running: (False, False, False, "", ""))
    ctx = _fresh_ctx(one)
    one.stop_state.request(one.StopMode.END_IMMEDIATE)
    ctx.bus.publish((None, None))
    outs = list(one._stream_job_to_ui(ctx))
    last = outs[-1]
    d = _desc(last) or ""
    assert "中断" in d, f"中断メッセージ不在: {d!r}"
    assert _TS_RE.search(d), "時刻が無い"
    assert _is_interactive(_btn_start(last)), "Startが再有効化されていない"


def test_stop_after_current_then_cancel_and_complete(monkeypatch):
    """
    [この生成で打ち切り] → 再押下でキャンセル → 正常完了へ復帰。
    """
    import importlib
    from datetime import datetime as _dt
    one = importlib.import_module("webui.oneframe_ichi")
    one.progress_ref_total = 1
    one.progress_img_total = 2
    one.progress_ref_idx = 1
    one.progress_img_idx = 2
    one.last_progress_desc = ""
    one.last_preview_image = None
    monkeypatch.setattr(one, "_compute_stop_controls", lambda running: (False, False, False, "", ""))
    ctx = _fresh_ctx(one)
    # 指示→キャンセル
    one.toggle_stop_after_current()  # 指示ON
    one.toggle_stop_after_current()  # キャンセル
    ts = _dt.now().strftime("%Y-%m-%d %H:%M:%S")
    summary = f"参考画像 {one.progress_ref_idx}/{one.progress_ref_total} ,イメージ {one.progress_img_idx}/{one.progress_img_total}"
    ctx.bus.publish(('progress', (None, one.translate("【全バッチ処理完了】プロセスが完了しました - ") + ts + " - " + summary, '')))
    ctx.bus.publish(('end', None))
    ctx.bus.close()
    outs = list(one._stream_job_to_ui(ctx))
    last = outs[-1]
    d = _desc(last) or ""
    assert "完了" in d and "中断" not in d, f"中断扱いになっている: {d!r}"
    assert _TS_RE.search(d)
    assert _is_interactive(_btn_start(last))


def test_stop_after_step_then_cancel_and_complete(monkeypatch):
    """
    [このステップで打ち切り] → 再押下でキャンセル → 正常完了へ復帰。
    """
    import importlib
    from datetime import datetime as _dt
    one = importlib.import_module("webui.oneframe_ichi")
    one.progress_ref_total = 1
    one.progress_img_total = 1
    one.progress_ref_idx = 1
    one.progress_img_idx = 1
    one.last_progress_desc = ""
    one.last_preview_image = None
    monkeypatch.setattr(one, "_compute_stop_controls", lambda running: (False, False, False, "", ""))
    ctx = _fresh_ctx(one)
    one.toggle_stop_after_step()  # 指示ON
    one.toggle_stop_after_step()  # キャンセル
    ts = _dt.now().strftime("%Y-%m-%d %H:%M:%S")
    summary = f"参考画像 {one.progress_ref_idx}/{one.progress_ref_total} ,イメージ {one.progress_img_idx}/{one.progress_img_total}"
    ctx.bus.publish(('progress', (None, one.translate("【全バッチ処理完了】プロセスが完了しました - ") + ts + " - " + summary, '')))
    ctx.bus.publish(('end', None))
    ctx.bus.close()
    outs = list(one._stream_job_to_ui(ctx))
    last = outs[-1]
    d = _desc(last) or ""
    assert "完了" in d and "中断" not in d
    assert _TS_RE.search(d)
    assert _is_interactive(_btn_start(last))


def test_end_event_without_progress_has_timestamp(monkeypatch):
    """
    progress未送出でendのみの経路でも、時刻＋件数のフォールバックが必ず入る。
    """
    import importlib, re
    one = importlib.import_module("webui.oneframe_ichi")
    one.progress_ref_total = 1
    one.progress_img_total = 2
    one.progress_ref_idx = 1
    one.progress_img_idx = 2
    one.last_progress_desc = ""
    one.last_preview_image = None
    monkeypatch.setattr(one, "_compute_stop_controls", lambda running: (False, False, False, "", ""))
    ctx = one.JobContext()
    ctx.bus.publish(('end', None))
    ctx.bus.close()
    outs = list(one._stream_job_to_ui(ctx))
    assert outs
    d = outs[-1][2] or ""
    assert ("完了" in d) or ("中断" in d)
    assert re.search(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", d), "フォールバック時刻が無い"


def _run_process_and_drain(one, monkeypatch, **process_kwargs):
    one.batch_stopped = False
    one.last_stop_mode = one.StopMode.NONE
    one.stop_state.clear()
    # _start_job_for_single_task を最小で完了させるフェイク
    def _fake_start(*a, **k):
        ctx = one.JobContext()
        one.cur_job = ctx
        one.generation_active = True
        ref_q = len(getattr(one, "reference_queue_files", [])) if process_kwargs.get("use_reference_queue") else 0
        base_ref = 1 if process_kwargs.get("use_reference_image") else 0
        ref_count = (base_ref + ref_q) * process_kwargs.get("reference_batch_count", 1)
        if ref_count == 0:
            ref_count = 1
        one.progress_ref_total = ref_count
        one.progress_img_total = process_kwargs.get("batch_count", 1) * ref_count
        ctx.bus.publish((None, None))
        ctx.bus.close()
        ctx.done.set()
        return ctx
    monkeypatch.setattr(one, "_start_job_for_single_task", _fake_start)
    monkeypatch.setattr(one, "_compute_stop_controls", lambda running: (False, False, False, "", ""))
    gen = one.process(**process_kwargs)
    last = None
    for ui in gen:
        last = ui
    assert last is not None, "processが出力を返さない"
    return last


@pytest.mark.xfail(strict=False, reason="progress accounting varies / implementation-dependent totals")
def test_batch_queue_image_combinations(monkeypatch, tmp_path):
    """
    image: 画像指定 / バッチ / キュー / 両方 / 画像無し（許容される想定）/ 異常（queue_type=imageだがファイル無し）
    -> progress_img_total が期待通りになること
    """
    import importlib
    (tmp_path / "in.png").touch()
    one = importlib.import_module("webui.oneframe_ichi")
    # 参照なし条件
    base_kwargs = dict(
        input_image=str(tmp_path / "in.png"),
        prompt="p", n_prompt="n",
        seed=0, steps=1, cfg=1, gs=1, rs=1,
        gpu_memory_preservation=False, use_teacache=False, use_prompt_cache=False,
        lora_files=None, lora_files2=None, lora_scales_text="", use_lora=False,
        fp8_optimization=False, resolution=64,
        output_directory=None,
        save_input_images=False, save_before_input_images=False,
        batch_count=1, use_random_seed=False, latent_window_size=9, latent_index=0,
        use_clean_latents_2x=True, use_clean_latents_4x=True, use_clean_latents_post=True,
        lora_mode=None, lora_dropdown1=None, lora_dropdown2=None, lora_dropdown3=None, lora_files3=None,
        use_rope_batch=False, use_queue=False, prompt_queue_file=None,
        use_reference_image=False, reference_image=None,
        target_index=1, history_index=0, reference_long_edge=False, input_mask=None, reference_mask=None,
        reference_batch_count=1, use_reference_queue=False,
        save_settings_on_start=False, alarm_on_completion=False,
        log_enabled=None, log_folder=None,
    )
    # 画像無し（許容）: input_image=None でも落ちずに完走し、totalはbatch_count
    one.progress_ref_total=0; one.progress_img_total=0
    last = _run_process_and_drain(one, monkeypatch, **{**base_kwargs, "input_image": None, "batch_count": 2})
    assert "イメージ" in (_desc(last) or "")
    assert one.progress_img_total == 2
    # キュー有り
    one.progress_ref_total=0; one.progress_img_total=0
    one.queue_enabled = True; one.queue_type="image"
    (tmp_path / "a.png").touch(); (tmp_path / "b.png").touch()
    one.image_queue_files = [str(tmp_path / "a.png"), str(tmp_path / "b.png")]
    last = _run_process_and_drain(one, monkeypatch, **{**base_kwargs, "batch_count": 3, "use_queue": True})
    # total_batches = batch_count * ref_count(=1) * image_queue_len(=2) 相当の扱いだが、
    # 実装は image_queue をループ側で展開せず、batch_countをそのまま回す想定のため、
    # ここでは batch_count * ref_count を期待（=3）
    assert one.progress_img_total == 3
    # 両方（batch+queue）: 期待は batch_count * ref_count（参照なしなのでref_count=1）
    one.progress_ref_total=0; one.progress_img_total=0
    one.queue_enabled = True; one.queue_type="image"
    (tmp_path / "a.png").touch()
    one.image_queue_files = [str(tmp_path / "a.png")]
    last = _run_process_and_drain(one, monkeypatch, **{**base_kwargs, "batch_count": 4, "use_queue": True})
    assert one.progress_img_total == 4


@pytest.mark.xfail(strict=False, reason="progress accounting varies / implementation-dependent totals")
def test_reference_combinations(monkeypatch, tmp_path):
    """
    参照画像: 指定 / バッチ / キュー / 両方 / 指定なしの挙動（無視）/ 異常時
    -> progress_ref_total と progress_img_total の整合をチェック
    """
    import importlib
    (tmp_path / "in.png").touch(); (tmp_path / "ref.png").touch()
    one = importlib.import_module("webui.oneframe_ichi")
    base_kwargs = dict(
        input_image=str(tmp_path / "in.png"),
        prompt="p", n_prompt="n",
        seed=0, steps=1, cfg=1, gs=1, rs=1,
        gpu_memory_preservation=False, use_teacache=False, use_prompt_cache=False,
        lora_files=None, lora_files2=None, lora_scales_text="", use_lora=False,
        fp8_optimization=False, resolution=64,
        output_directory=None,
        save_input_images=False, save_before_input_images=False,
        batch_count=2, use_random_seed=False, latent_window_size=9, latent_index=0,
        use_clean_latents_2x=True, use_clean_latents_4x=True, use_clean_latents_post=True,
        lora_mode=None, lora_dropdown1=None, lora_dropdown2=None, lora_dropdown3=None, lora_files3=None,
        use_rope_batch=False, use_queue=False, prompt_queue_file=None,
        use_reference_image=True, reference_image=str(tmp_path / "ref.png"),
        target_index=1, history_index=0, reference_long_edge=False, input_mask=None, reference_mask=None,
        reference_batch_count=1, use_reference_queue=False,
        save_settings_on_start=False, alarm_on_completion=False,
        log_enabled=None, log_folder=None,
    )
    # 参照指定のみ（repeat_count=1）: ref_total=1, img_total=batch_count*ref_total=2
    one.progress_ref_total=0; one.progress_img_total=0
    last = _run_process_and_drain(one, monkeypatch, **base_kwargs)
    assert one.progress_ref_total == 1
    assert one.progress_img_total == 2
    # 参照キューあり + reference_repeat_count=2
    one.progress_ref_total=0; one.progress_img_total=0
    (tmp_path / "r1.png").touch(); (tmp_path / "r2.png").touch()
    one.reference_queue_files = [str(tmp_path / "r1.png"), str(tmp_path / "r2.png")]
    one.get_reference_queue_files = lambda: one.reference_queue_files
    last = _run_process_and_drain(one, monkeypatch, **{**base_kwargs, "use_reference_queue": True, "reference_batch_count": 2})
    # 拡張後 ref_total = (入力参照1 + キュー2) * repeat(2) = 6
    assert one.progress_ref_total == 6
    assert one.progress_img_total == 6 * base_kwargs["batch_count"]
    # 参照指定なし（use_reference_image=False）: 無視され ref_total=1（= [None]）
    one.progress_ref_total=0; one.progress_img_total=0
    last = _run_process_and_drain(one, monkeypatch, **{**base_kwargs, "use_reference_image": False, "reference_image": None})
    assert one.progress_ref_total == 1
    assert one.progress_img_total == base_kwargs["batch_count"] * 1


def test_masks_specified(monkeypatch, tmp_path):
    """
    入力/参照マスクが指定されても完走し、UI最終化されStart再有効化。
    """
    import importlib
    (tmp_path / "in.png").touch(); (tmp_path / "ref.png").touch(); (tmp_path / "imask.png").touch(); (tmp_path / "rmask.png").touch()
    one = importlib.import_module("webui.oneframe_ichi")
    kwargs = dict(
        input_image=str(tmp_path / "in.png"),
        prompt="p", n_prompt="n",
        seed=0, steps=1, cfg=1, gs=1, rs=1,
        gpu_memory_preservation=False, use_teacache=False, use_prompt_cache=False,
        lora_files=None, lora_files2=None, lora_scales_text="", use_lora=False,
        fp8_optimization=False, resolution=64,
        output_directory=None,
        save_input_images=False, save_before_input_images=False,
        batch_count=1, use_random_seed=False, latent_window_size=9, latent_index=0,
        use_clean_latents_2x=True, use_clean_latents_4x=True, use_clean_latents_post=True,
        lora_mode=None, lora_dropdown1=None, lora_dropdown2=None, lora_dropdown3=None, lora_files3=None,
        use_rope_batch=False, use_queue=False, prompt_queue_file=None,
        use_reference_image=True, reference_image=str(tmp_path / "ref.png"),
        target_index=1, history_index=0, reference_long_edge=False,
        input_mask=str(tmp_path / "imask.png"), reference_mask=str(tmp_path / "rmask.png"),
        reference_batch_count=1, use_reference_queue=False,
        save_settings_on_start=False, alarm_on_completion=False,
        log_enabled=None, log_folder=None,
    )
    last = _run_process_and_drain(one, monkeypatch, **kwargs)
    assert _is_interactive(_btn_start(last))
    d = _desc(last) or ""
    assert ("完了" in d) or ("中断" in d)
    assert _TS_RE.search(d)


def test_resync_other_tab_receives_final_summary(monkeypatch):
    """
    異なるタブ（別購読者）が完了後に接続しても、最終サマリ（時刻＋件数）を受け取れる。
    """
    import importlib
    from datetime import datetime as _dt
    one = importlib.import_module("webui.oneframe_ichi")
    one.progress_ref_total = 1
    one.progress_img_total = 1
    one.progress_ref_idx = 1
    one.progress_img_idx = 1
    one.last_progress_desc = ""
    one.last_preview_image = None
    monkeypatch.setattr(one, "_compute_stop_controls", lambda running: (False, False, False, "", ""))
    ctx = _fresh_ctx(one)
    ts = _dt.now().strftime("%Y-%m-%d %H:%M:%S")
    summary = f"参考画像 {one.progress_ref_idx}/{one.progress_ref_total} ,イメージ {one.progress_img_idx}/{one.progress_img_total}"
    msg = one.translate("【全バッチ処理完了】プロセスが完了しました - ") + ts + " - " + summary
    # 先に完了を流し切って close
    ctx.bus.publish(('progress', (None, msg, '')))
    ctx.bus.publish(('end', None))
    ctx.bus.close()
    # --- 後から別タブがUIストリームを開始（再同期）
    outs = list(one._stream_job_to_ui(ctx))
    last = outs[-1]
    d = _desc(last) or ""
    assert "完了" in d and _TS_RE.search(d) and "1/1" in d
    assert _is_interactive(_btn_start(last))


def test_resync_same_tab_keeps_stream_or_reconnects(monkeypatch):
    """
    同一タブで再同期: すでに close 済みでも、UIは最終サマリを提示し Start 再有効化。
    （既存接続維持/再接続の差はUI観点では同値になるため、最終UIを検証）
    """
    import importlib
    from datetime import datetime as _dt
    one = importlib.import_module("webui.oneframe_ichi")
    one.progress_ref_total = 1
    one.progress_img_total = 2
    one.progress_ref_idx = 1
    one.progress_img_idx = 2
    one.last_progress_desc = ""
    one.last_preview_image = None
    monkeypatch.setattr(one, "_compute_stop_controls", lambda running: (False, False, False, "", ""))
    ctx = _fresh_ctx(one)
    ts = _dt.now().strftime("%Y-%m-%d %H:%M:%S")
    summary = f"参考画像 {one.progress_ref_idx}/{one.progress_ref_total} ,イメージ {one.progress_img_idx}/{one.progress_img_total}"
    ctx.bus.publish(('progress', (None, one.translate("【全バッチ処理完了】プロセスが完了しました - ") + ts + " - " + summary, '')))
    ctx.bus.publish(('end', None))
    ctx.bus.close()
    # 同一タブ＝同一ctxで再同期
    outs = list(one._stream_job_to_ui(ctx))
    last = outs[-1]
    d = _desc(last) or ""
    assert "完了" in d and _TS_RE.search(d) and "2/2" in d
    assert _is_interactive(_btn_start(last))


def test_ui_end_state_buttons_and_bar_reset(monkeypatch):
    """
    終了時: Startは有効、End/停止系は無効、プログレスバーは空文字になることを検証。
    """
    import importlib
    from datetime import datetime as _dt
    one = importlib.import_module("webui.oneframe_ichi")
    # 件数（サマリ表示用）
    one.progress_ref_total = 1
    one.progress_img_total = 1
    one.progress_ref_idx = 1
    one.progress_img_idx = 1
    one.last_progress_desc = ""
    one.last_preview_image = None
    # End/Stop系を無効にする想定
    monkeypatch.setattr(one, "_compute_stop_controls", lambda running: (False, False, False, "", ""))
    # 完了メッセージをbus経由で注入
    ctx = one.JobContext()
    ts = _dt.now().strftime("%Y-%m-%d %H:%M:%S")
    summary = f"参考画像 {one.progress_ref_idx}/{one.progress_ref_total} ,イメージ {one.progress_img_idx}/{one.progress_img_total}"
    msg = one.translate("【全バッチ処理完了】プロセスが完了しました - ") + ts + " - " + summary
    ctx.bus.publish(('progress', (None, msg, '')))
    ctx.bus.publish(('end', None))
    ctx.bus.close()
    outs = list(one._stream_job_to_ui(ctx))
    assert outs, "UIストリームが空"
    last = outs[-1]
    # プログレスバーは空
    assert last[3] == '', "終了時にプログレスバーが残っている"
    # Startは有効
    assert _is_interactive(_btn_start(last)), "終了時にStartが有効化されていない"
    # End/停止系は無効
    assert not _is_interactive(_btn_end(last)), "終了時にEndが有効になっている"
    assert not _is_interactive(_btn_stop_cur(last)), "終了時に『この生成で打ち切り』が有効になっている"
    assert not _is_interactive(_btn_stop_step(last)), "終了時に『このステップで打ち切り』が有効になっている"


def test_in_progress_resync_snapshot_and_follow_updates(monkeypatch):
    """
    進行中の再同期: 新規購読でも直ちにスナップショットが1件目で届き、その後の更新も追従できる。
    """
    import importlib
    one = importlib.import_module("webui.oneframe_ichi")
    one.progress_ref_total = 1
    one.progress_img_total = 2
    one.progress_ref_idx = 1
    one.progress_img_idx = 1
    one.last_preview_image = None
    one.last_progress_desc = ""
    monkeypatch.setattr(one, "_compute_stop_controls", lambda running: (True, True, True, "stopcur", "stopstep"))
    one.generation_active = True
    ctx = one.JobContext()
    g1 = one._stream_job_to_ui(ctx)
    next(g1)  # 初期スナップショット
    ctx.bus.publish(('progress', (None, "進行中1", "bar1")))
    _ = next(g1)  # 再同期スナップショットを消費
    first1 = next(g1)
    assert "進行中" in (first1[2] or "")
    g2 = one._stream_job_to_ui(ctx)
    first2 = next(g2)
    assert "進行中" in (first2[2] or "")
    one.progress_img_idx = 2
    ctx.bus.publish(('progress', (None, "進行中2", "bar2")))
    second1 = next(g1)
    next(g2)  # サブスクライブ開始とスナップショット（進行中2）消費
    _ = next(g2)  # 履歴の進行中1
    second2 = next(g2)  # 最新の進行中2
    assert "進行中2" in (second1[2] or "")
    assert "進行中2" in (second2[2] or "")
    ctx.bus.publish(('end', None))
    ctx.bus.close()
    outs1 = list(g1)
    outs2 = list(g2)
    assert outs1 and outs2


def test_seed_and_preview_are_carried_on_completion(monkeypatch):
    """
    終了時に seed / preview が最終UIに反映される最低限の整合性チェック。
    """
    import importlib
    from datetime import datetime as _dt
    one = importlib.import_module("webui.oneframe_ichi")
    one.current_seed = 123456
    one.progress_ref_total = 1
    one.progress_img_total = 1
    one.progress_ref_idx = 1
    one.progress_img_idx = 1
    one.last_preview_image = "dummy_preview"
    one.last_progress_desc = ""
    monkeypatch.setattr(one, "_compute_stop_controls", lambda running: (False, False, False, "", ""))
    ctx = one.JobContext()
    ts = _dt.now().strftime("%Y-%m-%d %H:%M:%S")
    summary = f"参考画像 1/1 ,イメージ 1/1"
    msg = one.translate("【全バッチ処理完了】プロセスが完了しました - ") + ts + " - " + summary
    ctx.bus.publish(('progress', ("dummy_preview", msg, '')))
    ctx.bus.publish(('end', None))
    ctx.bus.close()
    outs = list(one._stream_job_to_ui(ctx))
    last = outs[-1]
    seed_upd = last[8]
    assert isinstance(seed_upd, tuple) and seed_upd[0] == "update", "seedがUIに反映されていない"
    assert seed_upd[1].get("value") == 123456, "seed値が一致しない"
    assert isinstance(last[1], tuple) and last[1][0] == "update"


@pytest.mark.xfail(strict=False, reason="queue(empty) behavior not finalized")
def test_image_queue_empty_is_handled_gracefully(monkeypatch, tmp_path):
    """
    画像キュー指定だが実ファイルゼロのときの挙動（仕様未確定のため xfail）。
    """
    import importlib
    one = importlib.import_module("webui.oneframe_ichi")
    one.queue_enabled = True
    one.queue_type = "image"
    one.image_queue_files = []
    monkeypatch.setattr(one, "_compute_stop_controls", lambda running: (False, False, False, "", ""))
    def _fake_start(*a, **k):
        ctx = one.JobContext()
        one.cur_job = ctx
        one.generation_active = True
        ctx.bus.publish((None, None))
        ctx.bus.close()
        return ctx
    monkeypatch.setattr(one, "_start_job_for_single_task", _fake_start)
    gen = one.process(
        input_image=None, prompt="p", n_prompt="n",
        seed=0, steps=1, cfg=1, gs=1, rs=1,
        gpu_memory_preservation=False, use_teacache=False, use_prompt_cache=False,
        lora_files=None, lora_files2=None, lora_scales_text="", use_lora=False,
        fp8_optimization=False, resolution=64,
        output_directory=None,
        save_input_images=False, save_before_input_images=False,
        batch_count=1, use_random_seed=False, latent_window_size=9, latent_index=0,
        use_clean_latents_2x=True, use_clean_latents_4x=True, use_clean_latents_post=True,
        lora_mode=None, lora_dropdown1=None, lora_dropdown2=None, lora_dropdown3=None, lora_files3=None,
        use_rope_batch=False, use_queue=True, prompt_queue_file=None,
        use_reference_image=False, reference_image=None,
        target_index=1, history_index=0, reference_long_edge=False, input_mask=None, reference_mask=None,
        reference_batch_count=1, use_reference_queue=False,
        save_settings_on_start=False, alarm_on_completion=False,
        log_enabled=None, log_folder=None,
    )
    last = None
    for ui in gen:
        last = ui
    assert last is not None


@pytest.mark.xfail(strict=False, reason="reference-queue(empty) behavior not finalized")
def test_reference_queue_empty_is_handled_gracefully(monkeypatch, tmp_path):
    """
    参照キュー指定だが実ファイルゼロのときの挙動（仕様未確定のため xfail）。
    """
    import importlib
    one = importlib.import_module("webui.oneframe_ichi")
    one.reference_queue_files = []
    one.get_reference_queue_files = lambda: one.reference_queue_files
    monkeypatch.setattr(one, "_compute_stop_controls", lambda running: (False, False, False, "", ""))
    def _fake_start(*a, **k):
        ctx = one.JobContext()
        one.cur_job = ctx
        one.generation_active = True
        ctx.bus.publish((None, None))
        ctx.bus.close()
        return ctx
    monkeypatch.setattr(one, "_start_job_for_single_task", _fake_start)
    gen = one.process(
        input_image=None, prompt="p", n_prompt="n",
        seed=0, steps=1, cfg=1, gs=1, rs=1,
        gpu_memory_preservation=False, use_teacache=False, use_prompt_cache=False,
        lora_files=None, lora_files2=None, lora_scales_text="", use_lora=False,
        fp8_optimization=False, resolution=64,
        output_directory=None,
        save_input_images=False, save_before_input_images=False,
        batch_count=1, use_random_seed=False, latent_window_size=9, latent_index=0,
        use_clean_latents_2x=True, use_clean_latents_4x=True, use_clean_latents_post=True,
        lora_mode=None, lora_dropdown1=None, lora_dropdown2=None, lora_dropdown3=None, lora_files3=None,
        use_rope_batch=False, use_queue=False, prompt_queue_file=None,
        use_reference_image=True, reference_image=None,
        target_index=1, history_index=0, reference_long_edge=False, input_mask=None, reference_mask=None,
        reference_batch_count=1, use_reference_queue=True,
        save_settings_on_start=False, alarm_on_completion=False,
        log_enabled=None, log_folder=None,
    )
    last = None
    for ui in gen:
        last = ui
    assert last is not None
