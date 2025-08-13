# -*- coding: utf-8 -*-
import os, sys, re

sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))
sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/../webui"))
sys.argv = [sys.argv[0]]  # argparse対策

# 既存のスタブ群を取り込み（重い依存をダミー化）
import tests.smoke_stream_test  # noqa: F401


def _desc_from_ui(ui_tuple):
    # (_filename, _preview, desc, bar, start_btn, end_btn, stop_cur, stop_step, seed_upd)
    return ui_tuple[2]


def _start_enabled(ui_tuple):
    return ui_tuple[4][1].get("interactive", None) in (True, None)  # ("update", {"interactive":True,...})


def test_normal_completion_enables_start_and_has_timestamp(monkeypatch):
    import importlib

    one = importlib.import_module("webui.oneframe_ichi")

    # progress totals
    one.progress_ref_total = 1
    one.progress_img_total = 1
    one.progress_ref_idx = 1
    one.progress_img_idx = 1
    one.last_progress_desc = ""
    one.last_preview_image = None

    # 最小限のUIコントロール
    monkeypatch.setattr(one, "_compute_stop_controls", lambda running: (False, False, False, "", ""))
    monkeypatch.setattr(one, "ensure_dir", lambda p, name: p)
    monkeypatch.setattr(one, "get_output_folder_path", lambda p=None: os.getcwd())
    monkeypatch.setattr(one, "load_settings", lambda: {})
    monkeypatch.setattr(one, "save_settings", lambda *a, **k: True)

    def fake_start_job(*a, **k):
        ctx = one.JobContext()
        one.cur_job = ctx
        one.generation_active = True
        from datetime import datetime as _dt
        ts = _dt.now().strftime("%Y-%m-%d %H:%M:%S")
        summary = f"参考画像 {one.progress_ref_idx}/{one.progress_ref_total} ,イメージ {one.progress_img_idx}/{one.progress_img_total}"
        completion_message = one.translate("【全バッチ処理完了】プロセスが完了しました - ") + ts + " - " + summary
        ctx.bus.publish(('progress', (None, completion_message, '')))
        ctx.bus.publish(('end', None))
        ctx.bus.close()
        ctx.done.set()
        return ctx
    monkeypatch.setattr(one, "_start_job_for_single_task", fake_start_job)

    gen = one.process(
        None,
        "p",
        "n",
        0,
        1,
        1,
        1,
        1,
        False,
        False,
        False,
        None,
        None,
        "",
        False,
        False,
        64,
        None,
        False,
        False,
        batch_count=1,
        use_random_seed=False,
        latent_window_size=9,
        latent_index=0,
        use_clean_latents_2x=True,
        use_clean_latents_4x=True,
        use_clean_latents_post=True,
        lora_mode=None,
        lora_dropdown1=None,
        lora_dropdown2=None,
        lora_dropdown3=None,
        lora_files3=None,
        use_rope_batch=False,
        use_queue=False,
        prompt_queue_file=None,
        use_reference_image=False,
        reference_image=None,
        target_index=1,
        history_index=13,
        reference_long_edge=False,
        input_mask=None,
        reference_mask=None,
        reference_batch_count=1,
        use_reference_queue=False,
        save_settings_on_start=False,
        alarm_on_completion=False,
        log_enabled=None,
        log_folder=None,
    )
    last = None
    for ui in gen:
        last = ui
    assert last is not None, "process が何も返さず終了しました"
    desc = _desc_from_ui(last) or ""
    assert "完了" in desc or "中断" in desc
    assert re.search(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", desc), "完了メッセージに時刻がありません"
    assert _start_enabled(last), "完了後に Start が再有効化されていません"


def test_end_immediate_path_has_timestamp(monkeypatch):
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
    ctx.bus.publish((None, None))
    outs = list(one._stream_job_to_ui(ctx))
    last = outs[-1]
    desc = _desc_from_ui(last) or ""
    assert "中断" in desc
    assert re.search(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", desc), "中断メッセージに時刻がありません"
    assert _start_enabled(last), "中断後に Start が再有効化されていません"

