#!/bin/bash
echo "$(basename "$0") : 起動開始...."

# スクリプトのあるディレクトリを取得
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$ROOT_DIR/webui"

# Python実行パスの検出（優先度順）
if [ -f "$ROOT_DIR/venv/bin/python" ]; then
    # 仮想環境が存在する場合
    PYTHON_CMD="$ROOT_DIR/venv/bin/python"
elif command -v python3.10 &> /dev/null; then
    # python3.10が利用可能な場合
    PYTHON_CMD="python3.10"
elif command -v python3.9 &> /dev/null; then
    # python3.9が利用可能な場合
    PYTHON_CMD="python3.9"
elif command -v python3 &> /dev/null; then
    # python3が利用可能な場合
    PYTHON_CMD="python3"
else
    echo "エラー: Python 3.9以上が見つかりません"
    exit 1
fi

# Python環境の表示
echo "使用するPython環境: $PYTHON_CMD"
$PYTHON_CMD --version

# スクリプト実行
$PYTHON_CMD oneframe_ichi.py --lang ru