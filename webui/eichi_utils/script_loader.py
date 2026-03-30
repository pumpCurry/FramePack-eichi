"""
JS スクリプトローダー

webui/scripts/ フォルダ内の .js ファイルを自動検出し、
Gradio の head= パラメータ用の <script> タグを生成する。

使い方:
    from eichi_utils.script_loader import build_head_scripts

    head_html = build_head_scripts()
    block = gr.Blocks(css=css, head=head_html)

特徴:
    - scripts/ フォルダ内の .js ファイルをファイル名順で読み込み
    - インライン <script> として head に埋め込む
    - ユーザーが scripts/ に .js を追加するだけで自動読み込み
    - 読み込み順はファイル名のアルファベット順（00_xxx.js, 01_xxx.js で制御可能）
"""

import os
import glob


def get_scripts_dir() -> str:
    """scripts/ フォルダの絶対パスを返す。"""
    webui_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    return os.path.join(webui_dir, "scripts")


def find_js_files() -> list:
    """scripts/ フォルダ内の .js ファイルをファイル名順で返す。"""
    scripts_dir = get_scripts_dir()
    if not os.path.isdir(scripts_dir):
        return []
    pattern = os.path.join(scripts_dir, "*.js")
    files = sorted(glob.glob(pattern))
    return files


def build_head_scripts() -> str:
    """
    scripts/ フォルダ内の .js ファイルをインラインの <script> タグとして生成。

    Returns:
        str: <script> タグのHTML文字列。ファイルがなければ空文字列。
    """
    js_files = find_js_files()
    if not js_files:
        return ""

    tags = []
    for js_path in js_files:
        try:
            with open(js_path, encoding="utf-8") as f:
                content = f.read()
            tags.append(f"<script>\n{content}\n</script>")
        except Exception as e:
            print(f"[script_loader] Failed to read {js_path}: {e}")

    return "\n".join(tags)
