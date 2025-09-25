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

    /* 保存対象の設定項目を薄い青色でハイライト（ライト/ダークモード対応） */
    .saveable-setting {
        background-color: rgba(240, 248, 255, 0.5) !important; /* 薄い青色を透過指定（ライトモード） */
        border-left: 3px solid #90caf9 !important; /* 薄いボーダー色 */
    }
    
    /* システムのダークモード対応 */
    @media (prefers-color-scheme: dark) {
        .saveable-setting {
            background-color: rgba(25, 35, 60, 0.4) !important; /* ダークモードでの背景色 */
            border-left: 3px solid #64b5f6 !important; /* ダークモードでのボーダー色（少し明るめ） */
        }
    }
    
    /* Gradioのダークテーマ対応 */
    .dark .saveable-setting {
        background-color: rgba(25, 35, 60, 0.4) !important; /* ダークモードでの背景色 */
        border-left: 3px solid #64b5f6 !important; /* ダークモードでのボーダー色（少し明るめ） */
    }
    
    /* 保存対象項目のラベルにアイコンを追加 */
    .saveable-setting label::before {
        content: "💾 ";
        margin-right: 5px;
    }
    
    /* ダークモードでのラベル色調整 */
    .dark .saveable-setting label {
        color: #90caf9 !important; /* ダークモードで少し明るい青に */
    }

    /* markdownタイトル用 */
    .markdown-title {
        padding: 3px;
    }

    /* markdownサブタイトル用 */
    .markdown-subtitle {
        padding: 2px;
    }

    /* markdown領域用 */
    .markdown-desc {
        padding: 2px;
    }

    /* グルーピング用ボーダー */
    .group-border {
        border: solid 1px;
    }

    /* === Progress Row: 2行×3列のグリッド ===
       col1,row1–2: スピナー（rowspan=2）
       col2,row1   : グラフ（横いっぱい）
       col3,row1   : ％表示（右寄せ）
       col2–3,row2 : 状況説明（colspan=2）
    */
    .progress-grid {
      display: grid;
      grid-template-columns: auto 1fr auto; /* spinner | bar | percent */
      grid-template-rows: auto auto;        /* row1: bar/percent, row2: desc */
      gap: 8px;
      align-items: center;
    }

    /* スピナー: 2行ぶち抜き */
    .progress-spinner {
      grid-column: 1;
      grid-row: 1 / span 2;
    }

    /* 進捗グラフ（HTMLバー想定） */
    .progress-graph {
      grid-column: 2;
      grid-row: 1;
      min-width: 0; /* Grid内でのはみ出し抑止 */
    }

    /* グラフ内の要素が max-width で暴れないように */
    .progress-graph > * {
      width: 100%;
      max-width: 100% !important;
    }

    /* ％表示は右端へ寄せて右揃え */
    .progress-percent {
      grid-column: 3;
      grid-row: 1;
      justify-self: end;      /* グリッドセル内で右端 */
      text-align: right;
      white-space: nowrap;    /* 折り返し防止 */
      font-variant-numeric: tabular-nums; /* 数値の桁揃えがキレイに */
    }

    /* 状況説明は2列ぶち抜き */
    .progress-desc {
      grid-column: 2 / span 2;
      grid-row: 2;
    }

    /* スピナーの見た目が低くなる環境向けの保険（任意） */
    .progress-spinner .my-spinner,
    .progress-spinner .wrap, 
    .progress-spinner .loading {
      min-height: 28px;
    }

    """
