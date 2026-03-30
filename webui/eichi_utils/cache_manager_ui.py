"""
キャッシュ管理 UI パネル

Gradio の gr.Accordion にまとめたキャッシュ管理パネルを構築する。
build_cache_panel() は gr.Blocks() コンテキスト内で呼び出すこと。

削除時は confirm_modal.js のカスタムモーダルで確認後、
隠しボタン経由で実際の削除を実行する。
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
    """キャッシュ管理UIパネルを構築する。"""
    # 初期サイズ取得
    try:
        init_lora, init_prompt = _get_sizes(translate_fn)
    except Exception:
        init_lora = "- / -"
        init_prompt = "- / -"

    # 現在のフォーマット設定
    current_lora_fmt = lora_state_cache.get_preferred_format()
    current_fmt = current_lora_fmt

    with gr.Accordion(
        label=translate_fn("キャッシュ管理"),
        open=False,
    ) as accordion:
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown(f"### LoRA {translate_fn('キャッシュ')}")
                lora_size_md = gr.Markdown(value=init_lora, elem_id="eichi_lora_size_md")
            with gr.Column(scale=1):
                gr.Markdown(f"### {translate_fn('プロンプトキャッシュ')}")
                prompt_size_md = gr.Markdown(value=init_prompt, elem_id="eichi_prompt_size_md")

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

        # 隠し実行ボタン（モーダルの「承認して削除」からJSで呼ばれる）
        exec_lora_btn = gr.Button(visible=False, elem_id="eichi_exec_clear_lora")
        exec_prompt_btn = gr.Button(visible=False, elem_id="eichi_exec_clear_prompt")
        exec_all_btn = gr.Button(visible=False, elem_id="eichi_exec_clear_all")

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
        "exec_lora_btn": exec_lora_btn,
        "exec_prompt_btn": exec_prompt_btn,
        "exec_all_btn": exec_all_btn,
    }


def make_refresh_handler(translate_fn):
    """更新ボタンのハンドラを返す"""
    def handler():
        lora_text, prompt_text = _get_sizes(translate_fn)
        return lora_text, prompt_text, ""
    return handler


def make_exec_clear_lora_handler(translate_fn):
    """LoRAキャッシュの実際の削除ハンドラ（隠しボタンから呼ばれる）"""
    def handler():
        deleted, freed = cache_manager.clear_lora_cache(also_clear_inmem=True)
        freed_str = cache_manager.format_bytes(freed)
        lora_text, prompt_text = _get_sizes(translate_fn)
        status = f"✅ {translate_fn('削除完了')}: {deleted} {translate_fn('ファイル')} / {freed_str} {translate_fn('解放')}"
        return lora_text, prompt_text, status
    return handler


def make_exec_clear_prompt_handler(translate_fn):
    """プロンプトキャッシュの実際の削除ハンドラ（隠しボタンから呼ばれる）"""
    def handler():
        deleted, freed = cache_manager.clear_prompt_cache()
        freed_str = cache_manager.format_bytes(freed)
        lora_text, prompt_text = _get_sizes(translate_fn)
        status = f"✅ {translate_fn('削除完了')}: {deleted} {translate_fn('ファイル')} / {freed_str} {translate_fn('解放')}"
        return lora_text, prompt_text, status
    return handler


def make_exec_clear_all_handler(translate_fn):
    """全キャッシュの実際の削除ハンドラ（隠しボタンから呼ばれる）"""
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


# ---------------------------------------------------------------------------
# JS-driven modal helpers (Gradio js= parameter)
# innerHTML で <script> を注入しても実行されないため、
# Gradio の js= パラメータでブラウザ側から直接モーダルを呼ぶ。
# ---------------------------------------------------------------------------

def _esc_js(s):
    """JS文字列リテラル用エスケープ"""
    return str(s).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _build_confirm_js(translate_fn, target, message_key, exec_elem_id, detail_elem_id):
    """確認モーダルを表示する JS 関数文字列を生成する"""
    title = _esc_js(translate_fn("キャッシュ削除の確認"))
    message = _esc_js(translate_fn(message_key))
    warning = _esc_js(translate_fn("この操作は取り消せません"))
    confirm_label = _esc_js(f"⚠ {translate_fn('承認して削除')}")
    cancel_label = _esc_js(translate_fn("削除せず戻る"))

    # detail_elem_id が複数ある場合（all）はリストで受け取る
    detail_ids = detail_elem_id if isinstance(detail_elem_id, list) else [detail_elem_id]
    # .prose はGradioバージョンで存在しない場合がある。複数セレクタでフォールバック。
    # アコーディオンが閉じている場合は要素が未レンダリングなので "-" を返す。
    detail_js_parts = " + ' / ' + ".join(
        f'(function(){{ var e=document.querySelector("#{eid} .prose") || document.querySelector("#{eid} .markdown-text") || document.getElementById("{eid}"); return e ? e.textContent.trim() || "-" : "-"; }})()'
        for eid in detail_ids
    )

    return f"""() => {{
  var detail = {detail_js_parts};
  if (window._eichiConfirmModal) {{
    window._eichiConfirmModal({{
      title: "{title}",
      message: "{message}",
      detail: detail,
      warning: "{warning}",
      confirmLabel: "{confirm_label}",
      cancelLabel: "{cancel_label}",
      onConfirm: function() {{
        var btn = document.getElementById("{_esc_js(exec_elem_id)}");
        if (btn) btn.click();
      }}
    }});
  }} else {{
    if (confirm("{title}\\n{message}\\n" + detail)) {{
      var btn = document.getElementById("{_esc_js(exec_elem_id)}");
      if (btn) btn.click();
    }}
  }}
}}"""


def make_confirm_lora_js(translate_fn):
    """LoRAキャッシュ削除確認モーダルのJS文字列"""
    return _build_confirm_js(
        translate_fn, "lora",
        "LoRAキャッシュを削除します",
        "eichi_exec_clear_lora",
        "eichi_lora_size_md",
    )


def make_confirm_prompt_js(translate_fn):
    """プロンプトキャッシュ削除確認モーダルのJS文字列"""
    return _build_confirm_js(
        translate_fn, "prompt",
        "プロンプトキャッシュを削除します",
        "eichi_exec_clear_prompt",
        "eichi_prompt_size_md",
    )


def make_confirm_all_js(translate_fn):
    """全キャッシュ削除確認モーダルのJS文字列"""
    return _build_confirm_js(
        translate_fn, "all",
        "全キャッシュを削除します",
        "eichi_exec_clear_all",
        ["eichi_lora_size_md", "eichi_prompt_size_md"],
    )
