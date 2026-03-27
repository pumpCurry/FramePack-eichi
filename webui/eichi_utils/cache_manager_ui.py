"""
キャッシュ管理 UI パネル

Gradio の gr.Accordion にまとめたキャッシュ管理パネルを構築する。
build_cache_panel() は gr.Blocks() コンテキスト内で呼び出すこと。

使い方 (oneframe_ichi.py):
    from eichi_utils import cache_manager_ui as _cmu
    panel = _cmu.build_cache_panel(translate)
    # イベント接続は panel["refresh_btn"].click(...) 等で
"""

import gradio as gr
from eichi_utils import cache_manager
from eichi_utils import lora_state_cache
from eichi_utils import prompt_cache


def _size_text(entries, total_bytes, translate_fn):
    """サイズ表示用テキストを生成"""
    count = len(entries)
    size_str = cache_manager.format_bytes(total_bytes)
    formats = set(e["format"] for e in entries)
    fmt_str = "/".join(sorted(formats)) if formats else "-"
    return f"**{count}** {translate_fn('ファイル')} / **{size_str}** ({fmt_str})"


def _get_sizes(translate_fn):
    """両キャッシュのサイズ情報を取得"""
    lora_entries = cache_manager.lora_cache_entries()
    prompt_entries = cache_manager.prompt_cache_entries()
    lora_total = sum(e["size_bytes"] for e in lora_entries)
    prompt_total = sum(e["size_bytes"] for e in prompt_entries)
    lora_text = _size_text(lora_entries, lora_total, translate_fn)
    prompt_text = _size_text(prompt_entries, prompt_total, translate_fn)
    return lora_text, prompt_text


def build_cache_panel(translate_fn):
    """キャッシュ管理UIパネルを構築する。

    gr.Blocks() コンテキスト内で呼び出すこと。

    Returns:
        dict with Gradio component references
    """
    # 初期サイズ取得
    try:
        init_lora, init_prompt = _get_sizes(translate_fn)
    except Exception:
        init_lora = "- / -"
        init_prompt = "- / -"

    # 現在のフォーマット設定
    current_lora_fmt = lora_state_cache.get_preferred_format()
    current_prompt_fmt = prompt_cache.get_preferred_format()
    current_fmt = current_lora_fmt  # 両方同じにする前提

    with gr.Accordion(
        label=translate_fn("キャッシュ管理"),
        open=False,
    ) as accordion:
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown(f"### LoRA {translate_fn('キャッシュ')}")
                lora_size_md = gr.Markdown(value=init_lora)
            with gr.Column(scale=1):
                gr.Markdown(f"### {translate_fn('プロンプトキャッシュ')}")
                prompt_size_md = gr.Markdown(value=init_prompt)

        with gr.Row():
            cache_format_radio = gr.Radio(
                choices=["safetensors", "pt"],
                value=current_fmt,
                label=translate_fn("キャッシュ保存形式"),
                info=translate_fn("safetensors: 高速・安全 (推奨) / pt: レガシー互換"),
            )

        with gr.Row():
            refresh_btn = gr.Button(
                translate_fn("更新"),
                size="sm",
            )
            clear_lora_btn = gr.Button(
                f"🗑 LoRA {translate_fn('キャッシュを削除')}",
                variant="stop",
                size="sm",
            )
            clear_prompt_btn = gr.Button(
                f"🗑 {translate_fn('プロンプトキャッシュを削除')}",
                variant="stop",
                size="sm",
            )
            clear_all_btn = gr.Button(
                f"🗑 {translate_fn('全キャッシュを削除')}",
                variant="stop",
                size="sm",
            )

        status_md = gr.Markdown(value="")

    return {
        "accordion": accordion,
        "lora_size_md": lora_size_md,
        "prompt_size_md": prompt_size_md,
        "refresh_btn": refresh_btn,
        "clear_lora_btn": clear_lora_btn,
        "clear_prompt_btn": clear_prompt_btn,
        "clear_all_btn": clear_all_btn,
        "cache_format_radio": cache_format_radio,
        "status_md": status_md,
    }


def make_refresh_handler(translate_fn):
    """更新ボタンのハンドラを返す"""
    def handler():
        lora_text, prompt_text = _get_sizes(translate_fn)
        return lora_text, prompt_text, ""
    return handler


def make_clear_lora_handler(translate_fn):
    """LoRAキャッシュ削除ボタンのハンドラを返す"""
    def handler():
        deleted, freed = cache_manager.clear_lora_cache(also_clear_inmem=True)
        freed_str = cache_manager.format_bytes(freed)
        lora_text, prompt_text = _get_sizes(translate_fn)
        status = f"✅ {translate_fn('削除完了')}: {deleted} {translate_fn('ファイル')} / {freed_str} {translate_fn('解放')}"
        return lora_text, prompt_text, status
    return handler


def make_clear_prompt_handler(translate_fn):
    """プロンプトキャッシュ削除ボタンのハンドラを返す"""
    def handler():
        deleted, freed = cache_manager.clear_prompt_cache()
        freed_str = cache_manager.format_bytes(freed)
        lora_text, prompt_text = _get_sizes(translate_fn)
        status = f"✅ {translate_fn('削除完了')}: {deleted} {translate_fn('ファイル')} / {freed_str} {translate_fn('解放')}"
        return lora_text, prompt_text, status
    return handler


def make_clear_all_handler(translate_fn):
    """全キャッシュ削除ボタンのハンドラを返す"""
    def handler():
        result = cache_manager.clear_all_caches()
        lora_del, lora_freed = result["lora"]
        prompt_del, prompt_freed = result["prompt"]
        total_del = lora_del + prompt_del
        total_freed = lora_freed + prompt_freed
        freed_str = cache_manager.format_bytes(total_freed)
        lora_text, prompt_text = _get_sizes(translate_fn)
        status = f"✅ {translate_fn('削除完了')}: {total_del} {translate_fn('ファイル')} / {freed_str} {translate_fn('解放')}"
        return lora_text, prompt_text, status
    return handler


def make_format_change_handler(translate_fn):
    """保存形式変更ハンドラを返す"""
    def handler(fmt):
        fmt = str(fmt).strip().lower()
        if fmt not in ("safetensors", "pt"):
            fmt = "safetensors"
        lora_state_cache.set_preferred_format(fmt)
        prompt_cache.set_preferred_format(fmt)
        return f"✅ {translate_fn('保存形式を変更')}: {fmt}"
    return handler
