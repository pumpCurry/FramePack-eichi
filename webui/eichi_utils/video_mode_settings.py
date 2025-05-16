# -*- coding: utf-8 -*-
"""
FramePack-eichi 動画モード設定モジュール
柔軟なモード設定のための構造化データと関連ユーティリティ関数を提供します
"""

import math
import gradio as gr

from locales import i18n_extended
from locales.i18n_extended import translate

# モードタイプの定数定義
# 内部キーを使用
MODE_TYPE_NORMAL = translate("_KEY_MODE_NORMAL")
MODE_TYPE_LOOP = translate("_KEY_MODE_LOOP")

TRANSLATION_MAP = {
    # 英語の翻訳マッピング
    "1 second": "1秒",
    "2s": "2秒",
    "3s": "3秒",
    "4s": "4秒",
    "6s": "6秒",
    "8s": "8秒",
    "10s": "10秒",
    "12s": "12秒",
    "16s": "16秒",
    "20s": "20秒",
    # 中国語（繁体字）の翻訳マッピング
    "1秒": "1秒",
    "2秒": "2秒",
    "3秒": "3秒",
    "4秒": "4秒",
    "6秒": "6秒",
    "8秒": "8秒",
    "10秒": "10秒",
    "12秒": "12秒",
    "16秒": "16秒",
    "20秒": "20秒"
}

# ビデオモード設定の定義
# このデータ構造がモード選択の中心となります
VIDEO_MODE_SETTINGS = {
    "1秒": {
        "frames": 30,                   # 1秒×30FPS
        "sections": 1,                  # 必要セクション数（正確な計算に基づく）
        "display_seconds": 1.0,         # UI表示用秒数
        "important_keyframes": [0, 1],  # 重要なキーフレームのインデックス（0=赤枠、1=青枠）
        "keyframe_styles": {0: "red", 1: "blue"},  # キーフレームの枠線スタイル
        "copy_patterns": {
            MODE_TYPE_NORMAL: {
                "0": [],                # 短いためコピー不要
                "1": []                 # 短いためコピー不要
            },
            MODE_TYPE_LOOP: {
                "0": [],                # 短いためコピー不要
                "1": []                 # 短いためコピー不要
            }
        }
    },
    "1 second": {
        "frames": 30,                   # 1秒×30FPS
        "sections": 1,                  # 必要セクション数（正確な計算に基づく）
        "display_seconds": 1.0,         # UI表示用秒数
        "important_keyframes": [0, 1],  # 重要なキーフレームのインデックス（0=赤枠、1=青枠）
        "keyframe_styles": {0: "red", 1: "blue"},  # キーフレームの枠線スタイル
        "copy_patterns": {
            MODE_TYPE_NORMAL: {
                "0": [],                # 短いためコピー不要
                "1": []                 # 短いためコピー不要
            },
            MODE_TYPE_LOOP: {
                "0": [],                # 短いためコピー不要
                "1": []                 # 短いためコピー不要
            }
        }
    },
    "2秒": {
        "frames": 60,                   # 2秒×30FPS
        "sections": 2,                  # 必要セクション数（正確な計算に基づく）
        "display_seconds": 2.0,         # UI表示用秒数
        "important_keyframes": [0, 1],  # 重要なキーフレームのインデックス（0=赤枠、1=青枠）
        "keyframe_styles": {0: "red", 1: "blue"},  # キーフレームの枠線スタイル
        "copy_patterns": {
            MODE_TYPE_NORMAL: {
                "0": [],                # 短いためコピー不要
                "1": []                 # 短いためコピー不要
            },
            MODE_TYPE_LOOP: {
                "0": [],                # 短いためコピー不要
                "1": []                 # 短いためコピー不要
            }
        }
    },
    "3秒": {
        "frames": 90,                   # 3秒×30FPS
        "sections": 3,                  # 必要セクション数（正確な計算に基づく）
        "display_seconds": 3.0,         # UI表示用秒数
        "important_keyframes": [0, 1],  # 重要なキーフレームのインデックス（0=赤枠、1=青枠）
        "keyframe_styles": {0: "red", 1: "blue"},  # キーフレームの枠線スタイル
        "copy_patterns": {
            MODE_TYPE_NORMAL: {
                "0": [2],               # キーフレーム0→2にコピー（偶数パターン）
                "1": []                 # 奇数パターン（上限なし）
            },
            MODE_TYPE_LOOP: {
                "0": [2],               # キーフレーム0→2にコピー（偶数パターン）
                "1": []                 # 奇数パターン（上限なし）
            }
        }
    },
    "4秒": {
        "frames": 120,                  # 4秒×30FPS
        "sections": 4,                  # 必要セクション数（正確な計算に基づく）
        "display_seconds": 4.0,         # UI表示用秒数
        "important_keyframes": [0, 1],  # 重要なキーフレームのインデックス（0=赤枠、1=青枠）
        "keyframe_styles": {0: "red", 1: "blue"},  # キーフレームの枠線スタイル
        "copy_patterns": {
            MODE_TYPE_NORMAL: {
                "0": [2],               # キーフレーム0→2にコピー（偶数パターン）
                "1": [3]                # キーフレーム1→3にコピー（奇数パターン）
            },
            MODE_TYPE_LOOP: {
                "0": [2],               # キーフレーム0→2にコピー（偶数パターン）
                "1": [3]                # キーフレーム1→3にコピー（奇数パターン）
            }
        }
    },
    "6秒": {
        "frames": 180,                  # 6秒×30FPS
        "sections": 6,                  # 必要セクション数（正確な計算に基づく）
        "display_seconds": 6.0,         # UI表示用秒数
        "important_keyframes": [0, 1],  # 重要なキーフレームのインデックス（0=赤枠、1=青枠）
        "keyframe_styles": {0: "red", 1: "blue"},  # キーフレームの枠線スタイル
        "copy_patterns": {
            MODE_TYPE_NORMAL: {
                "0": [2, 4],            # キーフレーム0→2,4にコピー（偶数パターン）
                "1": [3, 5]             # キーフレーム1→3,5にコピー（奇数パターン）
            },
            MODE_TYPE_LOOP: {
                "0": [2, 4],            # キーフレーム0→2,4にコピー（偶数パターン）
                "1": [3, 5]             # キーフレーム1→3,5にコピー（奇数パターン）
            }
        }
    },
    "8秒": {
        "frames": 240,                  # 8秒×30FPS
        "sections": 8,                  # 必要セクション数
        "display_seconds": 8.0,         # UI表示用秒数
        "important_keyframes": [0, 1],  # 重要なキーフレームのインデックス（0=赤枠、1=青枠）
        "keyframe_styles": {0: "red", 1: "blue"},  # キーフレームの枠線スタイル
        "copy_patterns": {
            MODE_TYPE_NORMAL: {
                "0": [2, 4, 6],         # キーフレーム0→2,4,6にコピー（偶数パターン）
                "1": [3, 5, 7]          # キーフレーム1→3,5,7にコピー（奇数パターン）
            },
            MODE_TYPE_LOOP: {
                "0": [2, 4, 6],         # キーフレーム0→2,4,6にコピー（偶数パターン）
                "1": [3, 5, 7]          # キーフレーム1→3,5,7にコピー（奇数パターン）
            }
        }
    },
    "10秒": {
        "frames": 300,                  # 10秒×30FPS
        "sections": 10,                 # 必要セクション数
        "display_seconds": 10,        # UI表示用秒数
        "important_keyframes": [0, 1],  # 重要なキーフレームのインデックス（0=赤枠、1=青枠）
        "keyframe_styles": {0: "red", 1: "blue"},  # キーフレームの枠線スタイル
        "copy_patterns": {
            MODE_TYPE_NORMAL: {
                "0": [2, 4, 6, 8],      # キーフレーム0→2,4,6,8にコピー（偶数パターン）
                "1": [3, 5, 7, 9]       # キーフレーム1→3,5,7,9にコピー（奇数パターン）
            },
            MODE_TYPE_LOOP: {
                "0": [2, 4, 6, 8],      # キーフレーム0→2,4,6,8にコピー（偶数パターン）
                "1": [3, 5, 7, 9]       # キーフレーム1→3,5,7,9にコピー（奇数パターン）
            }
        }
    },
    "12秒": {
        "frames": 360,                  # 12秒×30FPS
        "sections": 12,                 # 必要セクション数
        "display_seconds": 12.0,        # UI表示用秒数
        "important_keyframes": [0, 1],  # 重要なキーフレームのインデックス（0=赤枠、1=青枠）
        "keyframe_styles": {0: "red", 1: "blue"},  # キーフレームの枠線スタイル
        "copy_patterns": {
            MODE_TYPE_NORMAL: {
                "0": [2, 4, 6, 8, 10],  # キーフレーム0→2,4,6,8,10にコピー（偶数パターン）
                "1": [3, 5, 7, 9, 11]   # キーフレーム1→3,5,7,9,11にコピー（奇数パターン）
            },
            MODE_TYPE_LOOP: {
                "0": [2, 4, 6, 8, 10],  # キーフレーム0→2,4,6,8,10にコピー（偶数パターン）
                "1": [3, 5, 7, 9, 11]   # キーフレーム1→3,5,7,9,11にコピー（奇数パターン）
            }
        }
    },
    "16秒": {
        "frames": 480,                  # 16秒×30FPS
        "sections": 15,                 # 必要セクション数（計算に基づく）
        "display_seconds": 16.0,        # UI表示用秒数
        "important_keyframes": [0, 1],  # 重要なキーフレームのインデックス（0=赤枠、1=青枠）
        "keyframe_styles": {0: "red", 1: "blue"},  # キーフレームの枠線スタイル
        "copy_patterns": {
            MODE_TYPE_NORMAL: {
                "0": [2, 4, 6, 8, 10, 12, 14],  # キーフレーム0→2,4,...,14にコピー（偶数パターン）
                "1": [3, 5, 7, 9, 11, 13]      # キーフレーム1→3,5,...,13にコピー（奇数パターン）
            },
            MODE_TYPE_LOOP: {
                "0": [2, 4, 6, 8, 10, 12, 14],  # キーフレーム0→2,4,...,14にコピー（偶数パターン）
                "1": [3, 5, 7, 9, 11, 13]      # キーフレーム1→3,5,...,13にコピー（奇数パターン）
            }
        }
    },
    "20秒": {
        "frames": 600,                  # 20秒×30FPS
        "sections": 19,                 # 必要セクション数（計算に基づく）
        "display_seconds": 20,        # UI表示用秒数
        "important_keyframes": [0, 1],  # 重要なキーフレームのインデックス（0=赤枠、1=青枠）
        "keyframe_styles": {0: "red", 1: "blue"},  # キーフレームの枠線スタイル
        "copy_patterns": {
            MODE_TYPE_NORMAL: {
                "0": [2, 4, 6, 8, 10, 12, 14, 16, 18],  # キーフレーム0→偶数番号にコピー
                "1": [3, 5, 7, 9, 11, 13, 15, 17]      # キーフレーム1→奇数番号にコピー
            },
            MODE_TYPE_LOOP: {
                "0": [2, 4, 6, 8, 10, 12, 14, 16, 18],  # キーフレーム0→偶数番号にコピー
                "1": [3, 5, 7, 9, 11, 13, 15, 17]      # キーフレーム1→奇数番号にコピー
            }
        }
    },
    "30秒": {
        "frames": 900,                  # 30秒×30FPS
        "sections": 28,                 # 必要セクション数（計算に基づく）
        "display_seconds": 30,          # UI表示用秒数
        "important_keyframes": [0, 1],  # 重要なキーフレームのインデックス（0=赤枠、1=青枠）
        "keyframe_styles": {0: "red", 1: "blue"},  # キーフレームの枠線スタイル
        "copy_patterns": {
            MODE_TYPE_NORMAL: {
                "0": [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26],  # キーフレーム0→偶数番号にコピー
                "1": [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27]   # キーフレーム1→奇数番号にコピー
            },
            MODE_TYPE_LOOP: {
                "0": [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26],  # キーフレーム0→偶数番号にコピー
                "1": [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27]   # キーフレーム1→奇数番号にコピー
            }
        }
    },
    "40秒": {
        "frames": 1200,                 # 40秒×30FPS
        "sections": 37,                 # 必要セクション数（計算に基づく）
        "display_seconds": 40,          # UI表示用秒数
        "important_keyframes": [0, 1],  # 重要なキーフレームのインデックス（0=赤枠、1=青枠）
        "keyframe_styles": {0: "red", 1: "blue"},  # キーフレームの枠線スタイル
        "copy_patterns": {
            MODE_TYPE_NORMAL: {
                "0": [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36],  # キーフレーム0→偶数番号にコピー
                "1": [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35]       # キーフレーム1→奇数番号にコピー
            },
            MODE_TYPE_LOOP: {
                "0": [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36],  # キーフレーム0→偶数番号にコピー
                "1": [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35]       # キーフレーム1→奇数番号にコピー
            }
        }
    },
    "60秒": {
        "frames": 1800,                 # 60秒×30FPS
        "sections": 55,                 # 必要セクション数（計算に基づく）
        "display_seconds": 60,          # UI表示用秒数
        "important_keyframes": [0, 1],  # 重要なキーフレームのインデックス（0=赤枠、1=青枠）
        "keyframe_styles": {0: "red", 1: "blue"},  # キーフレームの枠線スタイル
        "copy_patterns": {
            MODE_TYPE_NORMAL: {
                "0": [i for i in range(2, 55, 2)],  # キーフレーム0→偶数番号にコピー
                "1": [i for i in range(3, 55, 2)]   # キーフレーム1→奇数番号にコピー
            },
            MODE_TYPE_LOOP: {
                "0": [i for i in range(2, 55, 2)],  # キーフレーム0→偶数番号にコピー
                "1": [i for i in range(3, 55, 2)]   # キーフレーム1→奇数番号にコピー
            }
        }
    },
    "120秒": {
        "frames": 3600,                 # 120秒×30FPS
        "sections": 109,                # 必要セクション数（計算に基づく）
        "display_seconds": 120,         # UI表示用秒数
        "important_keyframes": [0, 1],  # 重要なキーフレームのインデックス（0=赤枠、1=青枠）
        "keyframe_styles": {0: "red", 1: "blue"},  # キーフレームの枠線スタイル
        "copy_patterns": {
            MODE_TYPE_NORMAL: {
                "0": [i for i in range(2, 109, 2)],  # キーフレーム0→偶数番号にコピー
                "1": [i for i in range(3, 109, 2)]   # キーフレーム1→奇数番号にコピー
            },
            MODE_TYPE_LOOP: {
                "0": [i for i in range(2, 109, 2)],  # キーフレーム0→偶数番号にコピー
                "1": [i for i in range(3, 109, 2)]   # キーフレーム1→奇数番号にコピー
            }
        }
    }
}

# HTMLキャッシュ関連
_html_cache = {}

def clear_html_cache():
    """HTMLキャッシュをクリアする"""
    global _html_cache
    _html_cache = {}

# ユーティリティ関数
def get_video_modes():
    """利用可能なビデオモードのリストを取得"""
    # 現在の言語に合わせてミックス値を生成する
    modes = []
    for key_suffix in ["1SEC", "2SEC", "3SEC", "4SEC", "6SEC", "8SEC", "10SEC", "12SEC", "16SEC", "20SEC", "30SEC", "40SEC", "60SEC", "120SEC"]:
        internal_key = f"_KEY_VIDEO_LENGTH_{key_suffix}"
        translated_value = translate(internal_key)
        modes.append(translated_value)
    return modes


def get_video_frames(mode_key):
    """モード名から総フレーム数を取得"""
    # 翻訳されたキーを元のキーに変換
    if mode_key in TRANSLATION_MAP:
        mode_key = TRANSLATION_MAP[mode_key]

    if mode_key not in VIDEO_MODE_SETTINGS:
        raise ValueError(translate("Unknown video mode: {mode_key}").format(mode_key=mode_key))
    return VIDEO_MODE_SETTINGS[mode_key]["frames"]


def get_video_seconds(mode_key):
    """モード名から表示用秒数を取得"""
    # 内部キーを使用したアプローチ
    # まず、mode_keyから内部キーを取得
    internal_key = i18n_extended.get_internal_key(mode_key)

    # 内部キーが_KEY_VIDEO_LENGTH_で始まる場合は、ストリップして秒数を取得
    if internal_key.startswith("_KEY_VIDEO_LENGTH_"):
        # 数字部分を取得 ("_KEY_VIDEO_LENGTH_10SEC" -> "10")
        seconds_str = internal_key.replace("_KEY_VIDEO_LENGTH_", "").replace("SEC", "")
        try:
            return float(seconds_str)
        except ValueError:
            pass

    # 当面は当初の方法も維持
    # 翻訳されたキーを元のキーに変換
    if mode_key in TRANSLATION_MAP:
        mode_key = TRANSLATION_MAP[mode_key]

    if mode_key not in VIDEO_MODE_SETTINGS:
        # 日本語キーからリテラルに秒数を取得する試み
        try:
            # 「秒」を削除して数値化
            seconds = float(mode_key.replace("秒", ""))
            return seconds
        except (ValueError, AttributeError):
            raise ValueError(translate("Unknown video mode: {mode_key}").format(mode_key=mode_key))

    return VIDEO_MODE_SETTINGS[mode_key]["display_seconds"]


def get_important_keyframes(mode_key):
    """重要なキーフレームのインデックスを取得"""
    # 内部キーを使用したアプローチ
    # まず、mode_keyから内部キーを取得してみる
    internal_key = i18n_extended.get_internal_key(mode_key)

    # 当面は当初の方法も維持
    # 内部キーが_KEY_VIDEO_LENGTH_で始まる場合は、ストリップして秒数を取得
    if internal_key.startswith("_KEY_VIDEO_LENGTH_"):
        # 数字部分を取得 ("_KEY_VIDEO_LENGTH_10SEC" -> "10")
        seconds_str = internal_key.replace("_KEY_VIDEO_LENGTH_", "").replace("SEC", "")
        try:
            seconds = float(seconds_str)
            # 秒数から日本語の動画モード名を生成
            japanese_mode_key = f"{int(seconds) if seconds.is_integer() else seconds}秒"
            if japanese_mode_key in VIDEO_MODE_SETTINGS:
                return VIDEO_MODE_SETTINGS[japanese_mode_key]["important_keyframes"]
            # デフォルトの重要キーフレームを返す
            return [0, 1]
        except ValueError:
            pass

    # 翻訳されたキーを元のキーに変換
    if mode_key in TRANSLATION_MAP:
        mode_key = TRANSLATION_MAP[mode_key]

    if mode_key not in VIDEO_MODE_SETTINGS:
        # 日本語の動画モードが見つからない場合は、デフォルトの重要キーフレームを返す
        try:
            # 「秒」を削除して数値化
            seconds = float(mode_key.replace("秒", ""))
            # デフォルトの重要キーフレームを返す
            return [0, 1]
        except (ValueError, AttributeError):
            raise ValueError(translate("Unknown video mode: {mode_key}").format(mode_key=mode_key))

    return VIDEO_MODE_SETTINGS[mode_key]["important_keyframes"]


def get_total_sections(mode_key):
    """モード名からセクション数を取得"""
    # 翻訳されたキーを元のキーに変換
    if mode_key in TRANSLATION_MAP:
        mode_key = TRANSLATION_MAP[mode_key]

    if mode_key not in VIDEO_MODE_SETTINGS:
        raise ValueError(translate("Unknown video mode: {mode_key}").format(mode_key=mode_key))
    return VIDEO_MODE_SETTINGS[mode_key]["sections"]


def get_copy_targets(mode, mode_key, keyframe_index, dynamic_sections=None):
    """指定キーフレームからのコピー先を取得

    実装では以下のパターンに対応:
    - セクション0(赤枠)から以降の偶数番号へのコピー
    - セクション1(青枠)から以降の奇数番号へのコピー
    """
    # 翻訳されたキーを元のキーに変換
    if mode_key in TRANSLATION_MAP:
        mode_key = TRANSLATION_MAP[mode_key]

    if mode_key not in VIDEO_MODE_SETTINGS:
        raise ValueError(translate("Unknown video mode: {mode_key}").format(mode_key=mode_key))

    # 必要なセクション数を取得 - 動的に計算された値を優先
    if dynamic_sections is not None:
        sections = dynamic_sections
        print(translate("[get_copy_targets] 動的に計算されたセクション数を使用: {sections}").format(sections=sections))
    else:
        # 設定からのデフォルト値を使用
        sections = VIDEO_MODE_SETTINGS[mode_key]["sections"]
        print(translate("[get_copy_targets] 設定値からのセクション数を使用: {sections}").format(sections=sections))

    # 常に動的なコピーパターンを生成
    dynamic_targets = []

    # セクション0(赤枠)からは以降の偶数番号にコピー
    if keyframe_index == 0:
        # 2から始まる偶数番号を表示中のセクションまでリストに追加
        dynamic_targets = [i for i in range(2, sections) if i % 2 == 0]
        print(translate("セクション0(赤枠)から偶数番号へのコピー: {dynamic_targets} (セクション数: {sections})").format(dynamic_targets=dynamic_targets, sections=sections))

    # セクション1(青枠)からは以降の奇数番号にコピー
    elif keyframe_index == 1:
        # 3から始まる奇数番号を表示中のセクションまでリストに追加
        dynamic_targets = [i for i in range(3, sections) if i % 2 == 1]
        print(translate("セクション1(青枠)から奇数番号へのコピー: {dynamic_targets} (セクション数: {sections})").format(dynamic_targets=dynamic_targets, sections=sections))

    return dynamic_targets


def get_max_keyframes_count():
    """設定で使用されている最大キーフレーム数を取得"""
    # 動的に計算するメソッド
    max_kf = 0
    for mode_key in VIDEO_MODE_SETTINGS:
        # まず重要なキーフレームの最大値をチェック
        if "important_keyframes" in VIDEO_MODE_SETTINGS[mode_key]:
            important_kfs = VIDEO_MODE_SETTINGS[mode_key]["important_keyframes"]
            if important_kfs and max(important_kfs) > max_kf:
                max_kf = max(important_kfs)

        # 次にコピーパターンの中の最大値をチェック
        # ループモード削除のためMODE_TYPE_NORMALのみをチェック
        for mode_type in [MODE_TYPE_NORMAL]:
            if mode_type not in VIDEO_MODE_SETTINGS[mode_key]["copy_patterns"]:
                continue

            for src_kf_str in VIDEO_MODE_SETTINGS[mode_key]["copy_patterns"][mode_type]:
                src_kf = int(src_kf_str)
                if src_kf > max_kf:
                    max_kf = src_kf

                targets = VIDEO_MODE_SETTINGS[mode_key]["copy_patterns"][mode_type][src_kf_str]
                if targets and max(targets) > max_kf:
                    max_kf = max(targets)

   # 計算されたセクション数の最大値も考慮
    for mode_key in VIDEO_MODE_SETTINGS:
        if "sections" in VIDEO_MODE_SETTINGS[mode_key]:
            sections = VIDEO_MODE_SETTINGS[mode_key]["sections"]
            if sections > max_kf:
                max_kf = sections

    # セクションの計算に必要な上限のみを返すように変更
    # 必要最小値として50を設定（UIの安定性のため）
    return max(max_kf + 1, 50)  # 0始まりなので+1、かつ最小50


def generate_keyframe_guide_html():
    """キーフレームガイドのHTML生成"""
    # キャッシュがあれば使用
    global _html_cache
    if "keyframe_guide" in _html_cache:
        return _html_cache["keyframe_guide"]

    html = f"""
    <div style="margin: 5px 0; padding: 10px; border-radius: 5px; border: 1px solid #ddd; background-color: #9a9a9a;">
        <span style="display: inline-block; margin-bottom: 5px; font-weight: bold;">{translate("■ キーフレーム画像設定ガイド:")}</span>
        <ul style="margin: 0; padding-left: 20px;">
    """

    # 新機能の説明を追加
    html += f"""
        <li><span style="color: #ff3860; font-weight: bold;">{translate("セクション0")}</span>: <span style="color: #ff3860; font-weight: bold;">{translate("赤枠")}</span>{translate("に画像を設定すると上限までの")}<span style="color: #ff3860; font-weight: bold;">{translate("すべての偶数番号セクション(0,2,4,6...)")}</span>{translate("に自動コピーされます")}</li>
        <li><span style="color: #3273dc; font-weight: bold;">{translate("セクション1")}</span>: <span style="color: #3273dc; font-weight: bold;">{translate("青枠")}</span>{translate("に画像を設定すると上限までの")}<span style="color: #3273dc; font-weight: bold;">{translate("すべての奇数番号セクション(1,3,5,7...)")}</span>{translate("に自動コピーされます")}</li>
        <li>{translate("「キーフレーム自動コピー機能を有効にする」チェックボックスで機能のオン/オフを切り替えられます")}</li>
    """

    # 各モードの説明を動的に生成
    for length, settings in VIDEO_MODE_SETTINGS.items():
        if length == translate("6秒") or length == translate("8秒"):
            continue  # 基本モードは説明不要

        # キーフレームを説明するためのリスト（スタイル別）
        keyframe_styles = settings.get("keyframe_styles", {})
        red_kfs = [kf+1 for kf, style in keyframe_styles.items() if style == "red"]  # 1始まりに変換
        blue_kfs = [kf+1 for kf, style in keyframe_styles.items() if style == "blue"]  # 1始まりに変換
        sections = settings.get("sections", 0)

        # 各モードのセクション数を表示
        html += translate('<li><span style="color: #333; font-weight: bold;">{length}</span> モード: 総セクション数 <b>{sections}</b>').format(length=length, sections=sections)

        # 偶数と奇数セクションの列挙
        even_sections = [i for i in range(sections) if i % 2 == 0]
        odd_sections = [i for i in range(sections) if i % 2 == 1]

        if even_sections:
            html += translate('<br>- <span style="color: #ff3860; font-weight: bold;">赤枠セクション0</span> → 偶数セクション {even_sections}').format(even_sections=even_sections)

        if odd_sections:
            html += translate('<br>- <span style="color: #3273dc; font-weight: bold;">青枠セクション1</span> → 奇数セクション {odd_sections}').format(odd_sections=odd_sections)

        html += '</li>'

    html += f"""
        {translate('<li><span style="color: #ff3860; font-weight: bold;">ループモード</span>では、常に<span style="color: #ff3860; font-weight: bold;">キーフレーム画像1(赤枠)</span>が重要です</li>')}
        </ul>
        <div style="margin-top: 5px; font-size: 0.9em; color: #666;">
            ※ {translate("全モードで拡張されたセクションコピー機能が動作します")}<br>
            ※ {translate("セクション0(赤枠)に画像を設定すると、すべての偶数セクション(0,2,4,6...)に自動適用されます")}<br>
            ※ {translate("セクション1(青枠)に画像を設定すると、すべての奇数セクション(1,3,5,7...)に自動適用されます")}<br>
            ※ {translate("0.5秒フレームサイズでも動的に正確なセクション数が計算されます")}<br>
            ※ {translate("セクション設定上部のチェックボックスでキーフレームコピー機能のオン/オフを切り替えられます")}
        </div>
    </div>
    """

    # キャッシュに保存
    _html_cache["keyframe_guide"] = html
    return html


# 動画モードの追加が容易になるようサポート関数を追加
def add_video_mode(mode_name, frames, sections, display_seconds, important_keyframes, copy_patterns, keyframe_styles=None):
    """
    新しい動画モードを設定に追加する関数

    Args:
        mode_name: モード名（例: "6秒"）
        frames: フレーム数
        sections: セクション数
        display_seconds: 表示用秒数
        important_keyframes: 重要なキーフレームのインデックス（0始まり）のリスト
        copy_patterns: コピーパターン辞書 {MODE_TYPE_NORMAL: {...}, MODE_TYPE_LOOP: {...}}
        keyframe_styles: キーフレームのスタイル辞書 {0: "red", 1: "blue", ...}（デフォルトはNone）
    """
    # デフォルトのスタイル設定
    if keyframe_styles is None:
        keyframe_styles = {kf: "red" for kf in important_keyframes}

    VIDEO_MODE_SETTINGS[mode_name] = {
        "frames": frames,
        "sections": sections,
        "display_seconds": display_seconds,
        "important_keyframes": important_keyframes,
        "keyframe_styles": keyframe_styles,
        "copy_patterns": copy_patterns
    }

    # ガイドHTML等のキャッシュをクリア
    clear_html_cache()


def handle_mode_length_change(mode, length, section_number_inputs):
    """モードと動画長の変更時のUI更新処理"""
    # 基本要素のクリア（Image, Final Frame）
    base_updates = [gr.update(value=None) for _ in range(2)]

    # キーフレーム画像の更新リスト生成
    keyframe_updates = []
    max_keyframes = get_max_keyframes_count()
    for i in range(max_keyframes):
        keyframe_updates.append(gr.update(value=None, elem_classes=""))

    # セクション番号ラベルのリセット
    for i in range(max_keyframes):
        if i < len(section_number_inputs):
            section_number_inputs[i].elem_classes = ""

    # 重要なキーフレームの強調表示
    important_kfs = get_important_keyframes(length)

    # キーフレームスタイルの取得（デフォルトは「red」）
    keyframe_styles = VIDEO_MODE_SETTINGS.get(length, {}).get("keyframe_styles", {})

    for idx in important_kfs:
        if idx < len(keyframe_updates):
            # スタイルを取得（デフォルトは「red」）
            style = keyframe_styles.get(idx, "red")
            style_class = f"highlighted-keyframe-{style}"

            keyframe_updates[idx] = gr.update(value=None, elem_classes=style_class)
            if idx < len(section_number_inputs):
                section_number_inputs[idx].elem_classes = f"highlighted-label-{style}"

    # ループモードの場合はキーフレーム0も重要（これはredスタイルを使用）
    if mode == MODE_TYPE_LOOP and 0 not in important_kfs:
        keyframe_updates[0] = gr.update(value=None, elem_classes="highlighted-keyframe-red")
        if 0 < len(section_number_inputs):
            section_number_inputs[0].elem_classes = "highlighted-label-red"

    # 動画長の設定
    video_length = get_video_seconds(length)

    # 結果を返す
    return base_updates + keyframe_updates + [gr.update(value=video_length)]


def process_keyframe_change(keyframe_idx, img, mode, length, enable_copy=True):
    """キーフレーム画像変更時の処理（汎用版）"""
    if img is None or not enable_copy:
        # 更新なし
        max_keyframes = get_max_keyframes_count()
        return [gr.update() for _ in range(max_keyframes - keyframe_idx - 1)]

    # コピー先の取得
    targets = get_copy_targets(mode, length, keyframe_idx)
    if not targets:
        max_keyframes = get_max_keyframes_count()
        return [gr.update() for _ in range(max_keyframes - keyframe_idx - 1)]

    # コピー先に対するアップデートを生成
    max_keyframes = get_max_keyframes_count()
    updates = []
    for i in range(keyframe_idx + 1, max_keyframes):
        # ターゲットインデックスへの相対位置を計算
        if (i - keyframe_idx) in targets:
            updates.append(gr.update(value=img))
        else:
            updates.append(gr.update())

    return updates


def print_settings_summary(enable_debug=False):
    """設定の概要をコンソールに出力（デバッグ用）"""
    if not enable_debug:
        return

    print("\n==== ビデオモード設定の概要 ====")
    for mode_key in VIDEO_MODE_SETTINGS:
        settings = VIDEO_MODE_SETTINGS[mode_key]
        print(translate("\nモード: {mode_key}").format(mode_key=mode_key))
        print(translate("  フレーム数: {frames}").format(frames=settings['frames']))
        print(translate("  セクション数: {sections}").format(sections=settings['sections']))
        print(translate("  表示秒数: {display_seconds}").format(display_seconds=settings['display_seconds']))
        print(translate("  重要キーフレーム: {important_keyframes}").format(important_keyframes=settings['important_keyframes']))
        print(translate("  コピーパターン:"))
        for mode_type in settings["copy_patterns"]:
            print("    {mode_type}:")
            for src, targets in settings["copy_patterns"][mode_type].items():
                print(translate("      キーフレーム{src} → {targets}").format(src=src, targets=targets))

    max_kf = get_max_keyframes_count()
    print(translate("\n最大キーフレーム数: {max_kf}").format(max_kf=max_kf))
    print("============================\n")


# 将来の拡張用に保持 - 現在は未使用
"""
def calculate_frames_per_section(latent_window_size=9):
    \"""1セクションあたりのフレーム数を計算\"""
    return latent_window_size * 4 - 3


def calculate_sections_from_frames(total_frames, latent_window_size=9):
    \"""フレーム数から必要なセクション数を計算\"""
    frames_per_section = calculate_frames_per_section(latent_window_size)
    return math.ceil(total_frames / frames_per_section)


def calculate_total_frame_count(sections, latent_window_size=9):
    \"""セクション数から総フレーム数を計算\"""
    frames_per_section = calculate_frames_per_section(latent_window_size)
    return sections * frames_per_section


def calculate_total_second_length(frames, fps=30):
    \"""フレーム数から秒数を計算\"""
    return frames / fps
"""