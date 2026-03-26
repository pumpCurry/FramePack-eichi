"""
共通プログレスバーモジュール

oneframe_ichi / endframe_ichi / endframe_ichi_f1 から共有される
テーマ対応・スピナー付きプログレスバーHTML生成。

使い方:
    from eichi_utils.progress_bar import make_progress_bar_html2

    html = make_progress_bar_html2(50, "[THEME=cyan]Encoding...")

テーマ色:
    yellow (#fbc02d) - 準備 / ロード
    cyan   (#4fc3f7) - エンコード (VAE, CLIP, テキスト)
    orange (#ff9800) - LoRA ロード / キャッシュ
    red    (#f44336) - キャッシュ書き出し / エラー
    green  (#4caf50) - 後処理 (保存, オフロード)
    blue   (#2222ff) - サンプリング (生成中)

タグ書式:
    [THEME=color]テキスト
    [THEME=color spinner=false]テキスト
    [BAR fg=#hex bg=#hex spinner=true]テキスト
"""

import html as _html_mod


# ---------------------------------------------------------------------------
#  テーマ色定義 (Material Design 近似)
# ---------------------------------------------------------------------------
THEME_COLORS = {
    "yellow": "#fbc02d",
    "orange": "#ff9800",
    "red":    "#f44336",
    "cyan":   "#4fc3f7",
    "green":  "#4caf50",
    "blue":   "#2222ff",
}


# ---------------------------------------------------------------------------
#  タグ解析
# ---------------------------------------------------------------------------
def parse_bar_tag(hint: str):
    """
    [THEME=orange] / [BAR fg=#f80 bg=#eee spinner=true] のタグを解析。
    戻り値: (theme:str|None, fg:str|None, bg:str|None, spinner:bool|None, text:str)
    ※ spinner は指定が無ければ None（呼び出し側の既定に委ねる）
    """
    theme = None
    fg = None
    bg = None
    spinner = None   # None=未指定, True/False=明示
    text = hint or ""
    try:
        if isinstance(hint, str) and hint.startswith("[") and "]" in hint:
            close = hint.find("]")
            head = hint[1:close].strip()
            text = hint[close + 1:].strip()
            if head.upper().startswith("THEME="):
                # "THEME=green spinner=false" → theme_val="green spinner=false"
                # スペースで分割して最初の部分だけをテーマ名として取得
                theme_val = head.split("=", 1)[1].strip().lower()
                theme = theme_val.split()[0] if theme_val else None
                parts = head.split()
                for p in parts[1:]:
                    if "=" in p:
                        k, v = p.split("=", 1)
                        k = k.strip().lower()
                        v = v.strip()
                        if k == "spinner":
                            spinner = (v.lower() in ("1", "true", "yes"))
            elif head.upper().startswith("BAR"):
                parts = head.split()
                for p in parts[1:]:
                    if "=" in p:
                        k, v = p.split("=", 1)
                        k = k.strip().lower()
                        v = v.strip()
                        if k == "fg":
                            fg = v
                        elif k == "bg":
                            bg = v
                        elif k == "spinner":
                            spinner = (v.lower() in ("1", "true", "yes"))
    except Exception:
        pass
    return theme, fg, bg, spinner, text


# ---------------------------------------------------------------------------
#  HTML 生成 (スピナー付き 2行テーブル)
# ---------------------------------------------------------------------------
def make_simple_bar(
    percent: int,
    fg_color: str,
    bg_color: str,
    text: str,
    spinner: bool = True,
) -> str:
    """
    インラインCSS の進捗バーHTMLを生成する。

    表示構造（2行テーブル）:
    ┌──────┬───────────────┬──────┐
    │ Spinner    │   ProgressBar                │        %値 │  ← 1行目
    │(rowspan=2) │   (横いっぱい)               │   (右寄せ) │
    │            ├───────────────┴──────┤
    │            │      説明文 (colspan=2)                    │  ← 2行目
    └──────┴──────────────────────┘
    spinner=False の場合は左列なし。
    """

    # ---- 入力の安全化 ----
    try:
        pct = int(percent if percent is not None else 0)
    except Exception:
        pct = 0
    pct = max(0, min(100, pct))

    fg = (fg_color or "#4fc3f7").strip()
    bg = (bg_color or "#eee").strip()

    try:
        text_esc = _html_mod.escape(text or "")
    except Exception:
        text_esc = text or ""

    # ---- colgroup ----
    if spinner:
        colgroup = (
            "<colgroup>"
            '  <col style="width:44px;">'
            '  <col>'
            '  <col style="width:56px;">'
            "</colgroup>"
        )
        desc_colspan = 2
    else:
        colgroup = (
            "<colgroup>"
            '  <col>'
            '  <col style="width:56px;">'
            "</colgroup>"
        )
        desc_colspan = 2

    # ---- spinner 列 ----
    spinner_td = ""
    if spinner:
        _vis = "hidden" if pct >= 100 else "visible"
        spinner_td = (
            '<td class="pc-spinner-cell" rowspan="2" '
            '    style="width:44px;padding:0;vertical-align:middle;text-align:left;border:0;outline:0;padding:10px 4px;">'
            f'  <div style="width:32px;height:32px;display:block;visibility:{_vis};margin:0;">'
            f'    <div class="loader" role="status" aria-live="polite" '
            f'         aria-hidden="{"true" if pct >= 100 else "false"}" '
            f'         style="display:inline-block;width:32px;height:32px;margin:0;"></div>'
            f'  </div>'
            '</td>'
        )

    # ---- 1行目: バー + % ----
    progress_html = (
        '<div class="pc-progress" role="progressbar" '
        '     aria-label="進捗" aria-valuemin="0" aria-valuemax="100" '
        f'     aria-valuenow="{pct}">'
        '  <div class="pc-progress__track" '
        '       style="position:relative;flex:1 1 auto;height:16px;'
        f'              border-radius:4px;background:{bg};overflow:hidden;">'
        '    <div class="pc-progress__bar" '
        f'         style="position:absolute;inset:0 auto 0 0;width:{pct}%;'
        f'                background:{fg};border-radius:4px;"></div>'
        '  </div>'
        '</div>'
    )

    row1 = (
        '<tr class="pc-progress-row">'
        f'{spinner_td}'
        '<td class="pc-progress-cell" '
        '    style="padding:4px 6px;border:0;outline:0;vertical-align:middle;">'
        f'  {progress_html}'
        '</td>'
        '<td class="pc-progress-percent" '
        '    style="padding:4px 6px;width:56px;border:0;outline:0;vertical-align:middle;">'
        '  <div style="text-align:right;white-space:nowrap;'
        '              font-variant-numeric:tabular-nums;font-weight:bold;">'
        f'    {pct}%'
        '  </div>'
        '</td>'
        '</tr>'
    )

    # ---- 2行目: テキスト ----
    meta_html = f'<div class="pc-progress__meta" style="font-size:12px;color:#6b7280;">{text_esc}</div>'
    row2 = (
        '<tr class="pc-progress-meta-row">'
        f'<td class="pc-progress-meta-cell" colspan="{desc_colspan}" '
        '    style="padding:2px 6px 6px;border:0;outline:0;">'
        f'  {meta_html}'
        '</td>'
        '</tr>'
    )

    html = (
        '<table class="pc-progress-table" border="0" cellpadding="0" cellspacing="0" '
        '       style="width:100%;border:0;outline:0;border-collapse:separate;border-spacing:0;table-layout:fixed;">'
        f'{colgroup}'
        f'{row1}'
        f'{row2}'
        '</table>'
    )
    return html


# ---------------------------------------------------------------------------
#  公開API: make_progress_bar_html2
# ---------------------------------------------------------------------------
def make_progress_bar_html2(percent, hint: str, spinner: bool = True):
    """
    テーマ/色/スピナーつきのバーHTMLを返す。

    - hint のタグで theme/fg/bg/spinner を上書き可能
    - テーマ/色が無くても spinner=True なら既定色で自前バーを返す
    - 例外時は旧 make_progress_bar_html にフォールバック
    """
    theme, fg, bg, spinner_tag, text = parse_bar_tag(hint)
    resolved_spinner = spinner_tag if (spinner_tag is not None) else bool(spinner)

    try:
        if theme in THEME_COLORS:
            return make_simple_bar(percent, THEME_COLORS[theme], (bg or "#eee"), text, spinner=resolved_spinner)
        if fg or bg:
            return make_simple_bar(percent, (fg or ""), (bg or ""), text, spinner=resolved_spinner)
        return make_simple_bar(percent, None, None, text, spinner=resolved_spinner)
    except Exception:
        pass

    # フォールバック: 旧バー
    try:
        from diffusers_helper.gradio.progress_bar import make_progress_bar_html
        return make_progress_bar_html(percent, text)
    except Exception:
        return make_simple_bar(percent, None, None, text, spinner=resolved_spinner)
