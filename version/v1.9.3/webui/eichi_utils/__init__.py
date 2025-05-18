"""
eichi_utils - FramePack-eichiのユーティリティモジュール
プリセット管理、設定管理、キーフレーム処理、動画モード設定などを含む
"""

__version__ = "1.0.0"

# 外部モジュールからアクセスできるようにエクスポート
from .vae_settings import (
    load_vae_settings,
    save_vae_settings,
    apply_vae_settings,
    create_vae_settings_ui,
    get_current_vae_settings_display
)
