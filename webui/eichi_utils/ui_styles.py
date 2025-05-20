"""
UI関連のスタイルを定義するモジュール
"""
from diffusers_helper.gradio.progress_bar import make_progress_bar_css

from locales.i18n import translate

def get_app_css():
    """
    アプリケーションのCSSスタイルを返す

    Returns:
        str: CSSスタイル定義
    """
    return make_progress_bar_css() + """
    .title-suffix {
        color: currentColor;
        opacity: 0.05;
    }

    /* 赤枠のキーフレーム - 偶数パターン用 */
    .highlighted-keyframe-red {
        border: 4px solid #ff3860 !important;
        box-shadow: 0 0 10px rgba(255, 56, 96, 0.5) !important;
        background-color: rgba(255, 56, 96, 0.05) !important;
        position: relative;
    }

    /* 赤枠キーフレームに「偶数番号」のラベルを追加 */
    .highlighted-keyframe-red::after {
    """ + 'content: "' + translate("偶数セクションのコピー元") + '"' + """;
        position: absolute;
        top: 5px;
        right: 5px;
        background: rgba(255, 56, 96, 0.8);
        color: white;
        padding: 2px 6px;
        font-size: 10px;
        border-radius: 4px;
        pointer-events: none;
    }

    /* 青枠のキーフレーム - 奇数パターン用 */
    .highlighted-keyframe-blue {
        border: 4px solid #3273dc !important;
        box-shadow: 0 0 10px rgba(50, 115, 220, 0.5) !important;
        background-color: rgba(50, 115, 220, 0.05) !important;
        position: relative;
    }

    /* 青枠キーフレームに「奇数番号」のラベルを追加 */
    .highlighted-keyframe-blue::after {
    """ + 'content: "' + translate("奇数セクションのコピー元") + '"' + """;
        position: absolute;
        top: 5px;
        right: 5px;
        background: rgba(50, 115, 220, 0.8);
        color: white;
        padding: 2px 6px;
        font-size: 10px;
        border-radius: 4px;
        pointer-events: none;
    }

    /* 引き続きサポート（古いクラス名）- 前方互換性用 */
    .highlighted-keyframe {
        border: 4px solid #ff3860 !important;
        box-shadow: 0 0 10px rgba(255, 56, 96, 0.5) !important;
        background-color: rgba(255, 56, 96, 0.05) !important;
    }

    /* 赤枠用セクション番号ラベル */
    .highlighted-label-red label {
        color: #ff3860 !important;
        font-weight: bold !important;
    }

    /* 青枠用セクション番号ラベル */
    .highlighted-label-blue label {
        color: #3273dc !important;
        font-weight: bold !important;
    }

    /* 引き続きサポート（古いクラス名）- 前方互換性用 */
    .highlighted-label label {
        color: #ff3860 !important;
        font-weight: bold !important;
    }

    /* オールパディングの高さ調整 */
    #all_padding_checkbox {
        padding-top: 1.5rem;
        min-height: 5.8rem;
    }

    #all_padding_checkbox .wrap {
        align-items: flex-start;
    }

    #all_padding_checkbox .label-wrap {
        margin-bottom: 0.8rem;
        font-weight: 500;
        font-size: 14px;
    }

    #all_padding_checkbox .info {
        margin-top: 0.2rem;
    }

    /* セクション間の区切り線を太くする */
    .section-row {
        border-bottom: 4px solid #3273dc;
        margin-bottom: 20px;
        padding-bottom: 15px;
        margin-top: 10px;
        position: relative;
    }

    /* セクション番号を目立たせる */
    .section-row .gr-form:first-child label {
        font-weight: bold;
        font-size: 1.1em;
        color: #3273dc;
        background-color: rgba(50, 115, 220, 0.1);
        padding: 5px 10px;
        border-radius: 4px;
        margin-bottom: 10px;
        display: inline-block;
    }

    /* セクションの背景を少し強調 */
    .section-row {
        background-color: rgba(50, 115, 220, 0.03);
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }

    /* セクション間の余白を増やす */
    .section-container > .gr-block:not(:first-child) {
        margin-top: 10px;
    }

    /* アコーディオンセクションのスタイル */
    .section-accordion {
        margin-top: 15px;
        margin-bottom: 15px;
        border-left: 4px solid #3273dc;
        padding-left: 10px;
    }

    .section-accordion h3 button {
        font-weight: bold;
        color: #3273dc;
    }

    .section-accordion .gr-block {
        border-radius: 8px;
    }

    /* 保存対象の設定項目を薄い青色でハイライト */
    .saveable-setting {
        background-color: #f0f8ff !important; /* より薄い背景色（AliceBlue） */
        border-left: 3px solid #90caf9 !important; /* より薄いボーダー色 */
    }
    
    /* 保存対象項目のラベルにアイコンを追加 */
    .saveable-setting label::before {
        content: "💾 ";
        margin-right: 5px;
    }
    """