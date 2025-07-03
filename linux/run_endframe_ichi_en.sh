#!/bin/bash
echo "$(basename "$0") : 起動開始...."
# English Endframe Execution Script
# Note: Unofficial support - no warranty

# Get the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# FramePack-eichi root directory (parent of script directory)
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

# Navigate to main FramePack directory
cd "$ROOT_DIR/webui/submodules/FramePack"
python3 webui/endframe_ichi.py --lang=en "$@"