"""
LoRAプリセット管理モジュール
LoRA選択とスケール値のプリセット保存・読み込み機能を提供
"""

import os
import json
import traceback
from datetime import datetime

from locales.i18n_extended import translate

def get_lora_presets_folder_path():
    """LoRAプリセットフォルダの絶対パスを取得する"""
    # eichi_utils直下からwebuiフォルダに移動し、presetsフォルダを使用
    webui_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(webui_path, 'presets')

def initialize_lora_presets():
    """初期LoRAプリセットファイルがない場合に作成する関数"""
    presets_folder = get_lora_presets_folder_path()
    os.makedirs(presets_folder, exist_ok=True)
    preset_file = os.path.join(presets_folder, 'lora_presets.json')

    # デフォルトのLoRA設定
    default_lora_configs = [
        {
            "name": translate("デフォルト設定1"),
            "lora1": translate("なし"),
            "lora2": translate("なし"),
            "lora3": translate("なし"),
            "scales": "0.8,0.8,0.8"
        },
        {
            "name": translate("デフォルト設定2"),
            "lora1": translate("なし"),
            "lora2": translate("なし"),
            "lora3": translate("なし"),
            "scales": "1.0,0.5,0.3"
        }
    ]

    # 既存ファイルがあり、正常に読み込める場合は終了
    if os.path.exists(preset_file):
        try:
            with open(preset_file, 'r', encoding='utf-8') as f:
                presets_data = json.load(f)
            return
        except:
            # エラーが発生した場合は新規作成
            pass

    # 新規作成
    presets_data = {
        "presets": [],
        "default_preset_index": 0
    }

    # デフォルトのプリセットを追加
    for i, config in enumerate(default_lora_configs):
        presets_data["presets"].append({
            "name": config["name"],
            "lora1": config["lora1"],
            "lora2": config["lora2"],
            "lora3": config["lora3"],
            "scales": config["scales"],
            "timestamp": datetime.now().isoformat(),
            "is_default": True
        })

    # 保存
    try:
        with open(preset_file, 'w', encoding='utf-8') as f:
            json.dump(presets_data, f, ensure_ascii=False, indent=2)
    except:
        # 保存に失敗してもエラーは出さない（次回起動時に再試行される）
        pass

def load_lora_presets():
    """LoRAプリセットを読み込む関数"""
    presets_folder = get_lora_presets_folder_path()
    os.makedirs(presets_folder, exist_ok=True)
    preset_file = os.path.join(presets_folder, 'lora_presets.json')
    
    # 初期化（必要に応じて）
    initialize_lora_presets()
    
    # プリセットファイルを読み込む
    try:
        with open(preset_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data["presets"], data.get("default_preset_index", 0)
    except:
        # エラーの場合は空のプリセットリストを返す
        return [], 0

def save_lora_preset(preset_index, lora1, lora2, lora3, scales):
    """LoRAプリセットを保存する関数"""
    presets_folder = get_lora_presets_folder_path()
    os.makedirs(presets_folder, exist_ok=True)
    preset_file = os.path.join(presets_folder, 'lora_presets.json')
    
    # 既存のプリセットを読み込む
    try:
        with open(preset_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except:
        data = {"presets": [], "default_preset_index": 0}
    
    # 5つのプリセットを確保
    while len(data["presets"]) < 5:
        data["presets"].append({
            "name": translate("設定{0}").format(len(data["presets"]) + 1),
            "lora1": translate("なし"),
            "lora2": translate("なし"),
            "lora3": translate("なし"),
            "scales": "0.8,0.8,0.8",
            "timestamp": datetime.now().isoformat(),
            "is_default": False
        })
    
    # 指定されたプリセットを更新
    if 0 <= preset_index < 5:
        data["presets"][preset_index] = {
            "name": translate("設定{0}").format(preset_index + 1),
            "lora1": lora1 or translate("なし"),
            "lora2": lora2 or translate("なし"),
            "lora3": lora3 or translate("なし"),
            "scales": scales or "0.8,0.8,0.8",
            "timestamp": datetime.now().isoformat(),
            "is_default": False
        }
        
        # 保存
        with open(preset_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            
        return True, translate("設定{0}を保存しました").format(preset_index + 1)
    else:
        return False, translate("無効なプリセット番号です")

def load_lora_preset(preset_index):
    """指定されたプリセットを読み込む関数"""
    presets, _ = load_lora_presets()
    
    if 0 <= preset_index < len(presets):
        preset = presets[preset_index]
        return (
            preset.get("lora1", translate("なし")),
            preset.get("lora2", translate("なし")),
            preset.get("lora3", translate("なし")),
            preset.get("scales", "0.8,0.8,0.8")
        )
    else:
        # デフォルト値を返す
        return None

def get_preset_names():
    """プリセット名のリストを取得する関数"""
    presets, _ = load_lora_presets()
    names = []
    for i in range(5):
        if i < len(presets):
            names.append(presets[i].get("name", translate("設定{0}").format(i + 1)))
        else:
            names.append(translate("設定{0}").format(i + 1))
    return names