"""
設定ファイル管理モジュール
endframe_ichi.pyから外出しした設定ファイル関連処理を含む
"""

import os
import json
import subprocess
from locales.i18n_extended import translate

def get_settings_file_path():
    """設定ファイルの絶対パスを取得する"""
    # eichi_utils直下からwebuiフォルダに移動
    webui_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    settings_folder = os.path.join(webui_path, 'settings')
    return os.path.join(settings_folder, 'app_settings.json')

def get_output_folder_path(folder_name=None):
    """出力フォルダの絶対パスを取得する"""
    # eichi_utils直下からwebuiフォルダに移動
    webui_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if not folder_name or not folder_name.strip():
        folder_name = "outputs"
    return os.path.join(webui_path, folder_name)

def initialize_settings():
    """設定ファイルを初期化する（存在しない場合のみ）"""
    settings_file = get_settings_file_path()
    settings_dir = os.path.dirname(settings_file)

    if not os.path.exists(settings_file):
        # 初期デフォルト設定（アプリケーション設定を含む）
        default_settings = {
            'output_folder': 'outputs',
            'app_settings_eichi': get_default_app_settings(),
            'log_settings': {'log_enabled': False, 'log_folder': 'logs'}
        }
        try:
            os.makedirs(settings_dir, exist_ok=True)
            with open(settings_file, 'w', encoding='utf-8') as f:
                json.dump(default_settings, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(translate("設定ファイル初期化エラー: {0}").format(e))
            return False
    return True

def load_settings():
    """設定を読み込む関数"""
    settings_file = get_settings_file_path()
    default_settings = {'output_folder': 'outputs'}

    if os.path.exists(settings_file):
        try:
            with open(settings_file, 'r', encoding='utf-8') as f:
                file_content = f.read()
                if not file_content.strip():
                    return default_settings
                settings = json.loads(file_content)

                # デフォルト値とマージ
                for key, value in default_settings.items():
                    if key not in settings:
                        settings[key] = value
                return settings
        except Exception as e:
            print(translate("設定読み込みエラー: {0}").format(e))

    return default_settings

def save_settings(settings):
    """設定を保存する関数"""
    settings_file = get_settings_file_path()

    try:
        # 保存前にディレクトリが存在するか確認
        os.makedirs(os.path.dirname(settings_file), exist_ok=True)

        # JSON書き込み
        with open(settings_file, 'w', encoding='utf-8') as f:
            json.dump(settings, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(translate("設定保存エラー: {0}").format(e))
        return False

def open_output_folder(folder_path):
    """指定されたフォルダをOSに依存せず開く"""
    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)

    try:
        if os.name == 'nt':  # Windows
            subprocess.Popen(['explorer', folder_path])
        elif os.name == 'posix':  # Linux/Mac
            try:
                subprocess.Popen(['xdg-open', folder_path])
            except:
                subprocess.Popen(['open', folder_path])
        print(translate("フォルダを開きました: {0}").format(folder_path))
        return True
    except Exception as e:
        print(translate("フォルダを開く際にエラーが発生しました: {0}").format(e))
        return False

def get_localized_default_value(key, current_lang="ja"):
    """言語に応じたデフォルト値を返す
    
    Args:
        key (str): 設定キー
        current_lang (str): 現在の言語コード
        
    Returns:
        適切な言語に翻訳されたデフォルト値
    """
    # 特別な翻訳が必要な値のマッピング
    localized_values = {
        "frame_save_mode": {
            "ja": "保存しない",
            "en": "Do not save",
            "zh-tw": "不保存",
            "ru": "Не сохранять"
        }
        # 必要に応じて他の値も追加可能
    }
    
    # キーが存在するか確認し、言語に応じた値を返す
    if key in localized_values:
        # 指定された言語が存在しない場合はjaをデフォルトとして使用
        return localized_values[key].get(current_lang, localized_values[key]["ja"])
    
    # 特別な翻訳が必要ない場合は None を返す
    return None

def get_default_app_settings(current_lang="ja"):
    """eichiのデフォルト設定を返す
    
    Args:
        current_lang (str, optional): 現在の言語コード. Defaults to "ja".
    """
    # フレーム保存モードのデフォルト値を言語に応じて設定
    frame_save_mode_default = get_localized_default_value("frame_save_mode", current_lang)
    
    return {
        # 基本設定
        "resolution": 640,
        "mp4_crf": 16,
        "steps": 25,
        "cfg": 1.0,
        
        # パフォーマンス設定
        "use_teacache": True,
        "gpu_memory_preservation": 6,
        "use_vae_cache": False,
        
        # 詳細設定
        "gs": 10.0,  # Distilled CFG Scale
        
        # パディング設定
        "use_all_padding": False,
        "all_padding_value": 1.0,
        
        # エンドフレーム設定
        "end_frame_strength": 1.0,
        
        # 保存設定
        "keep_section_videos": False,
        "save_section_frames": False,
        "save_tensor_data": False,
        "frame_save_mode": frame_save_mode_default if frame_save_mode_default else "保存しない",
        
        # 自動保存設定
        "save_settings_on_start": False,
        
        # アラーム設定
        "alarm_on_completion": True
    }

def get_default_app_settings_f1(current_lang="ja"):
    """F1のデフォルト設定を返す
    
    Args:
        current_lang (str, optional): 現在の言語コード. Defaults to "ja".
    """
    # フレーム保存モードのデフォルト値を言語に応じて設定
    frame_save_mode_default = get_localized_default_value("frame_save_mode", current_lang)
    
    return {
        # 基本設定
        "resolution": 640,
        "mp4_crf": 16,
        "steps": 25,
        "cfg": 2.5,
        
        # パフォーマンス設定
        "use_teacache": True,
        "gpu_memory_preservation": 6,
        
        # 詳細設定
        "gs": 10,
        
        # F1独自設定
        "image_strength": 1.0,
        
        # 保存設定
        "keep_section_videos": False,
        "save_section_frames": False,
        "save_tensor_data": False,
        "frame_save_mode": frame_save_mode_default if frame_save_mode_default else "保存しない",
        
        # 自動保存・アラーム設定
        "save_settings_on_start": False,
        "alarm_on_completion": True
    }

def get_default_app_settings_oichi():
    """oichiのデフォルト設定を返す"""
    return {
        # 基本設定
        "resolution": 640,
        "steps": 25,
        "cfg": 2.5,
        
        # パフォーマンス設定
        "use_teacache": True,
        "gpu_memory_preservation": 6,
        
        # 詳細設定
        "gs": 10,
        
        # oneframe固有設定
        "latent_window_size": 9,
        "latent_index": 0,
        "use_clean_latents_2x": True,
        "use_clean_latents_4x": True,
        "use_clean_latents_post": True,
        
        # インデックス設定
        "target_index": 1,
        "history_index": 16,
        
        # LoRA設定
        "use_lora": False,
        "lora_mode": "ディレクトリから選択",
        
        # 最適化設定
        "fp8_optimization": True,
        
        # バッチ設定
        "batch_count": 1,
        
        # RoPE設定
        "use_rope_batch": False,
        
        # キュー設定
        "use_queue": False,
        
        # 自動保存・アラーム設定
        "save_settings_on_start": False,
        "alarm_on_completion": True
    }

def load_app_settings():
    """eichiのアプリケーション設定を読み込む"""
    settings = load_settings()
    # 旧キーからの移行処理
    if 'app_settings' in settings and 'app_settings_eichi' not in settings:
        settings['app_settings_eichi'] = settings['app_settings']
        del settings['app_settings']
        save_settings(settings)
    elif 'app_settings_eichi' not in settings:
        settings['app_settings_eichi'] = get_default_app_settings()
        save_settings(settings)
    
    # 既存の設定にデフォルト値をマージ（新しいキーのため）
    app_settings = settings.get('app_settings_eichi', {})
    default_settings = get_default_app_settings()
    
    # 存在しないキーにはデフォルト値を使用
    for key, default_value in default_settings.items():
        if key not in app_settings:
            app_settings[key] = default_value
            print(f"[INFO] 新しい設定項目 '{key}' をデフォルト値 {default_value} で追加")
    
    # マージした設定を保存
    if app_settings != settings.get('app_settings_eichi', {}):
        settings['app_settings_eichi'] = app_settings
        save_settings(settings)
    
    return app_settings

def save_app_settings(app_settings):
    """eichiのアプリケーション設定を保存"""
    settings = load_settings()
    
    # 不要なキーを除外してコピー（手動保存と自動保存の一貫性のため）
    filtered_settings = {k: v for k, v in app_settings.items() 
                        if k not in ['rs', 'output_dir', 'frame_size_radio']}
    
    settings['app_settings_eichi'] = filtered_settings
    return save_settings(settings)

def load_app_settings_f1():
    """F1のアプリケーション設定を読み込む"""
    settings = load_settings()
    
    # F1の設定が存在しない場合はデフォルト値を設定
    if 'app_settings_f1' not in settings:
        settings['app_settings_f1'] = get_default_app_settings_f1()
        save_settings(settings)
    
    # 既存の設定にデフォルト値をマージ（新しいキーのため）
    app_settings = settings.get('app_settings_f1', {})
    default_settings = get_default_app_settings_f1()
    
    # 存在しないキーにはデフォルト値を使用
    for key, default_value in default_settings.items():
        if key not in app_settings:
            app_settings[key] = default_value
            print(f"[INFO] F1: 新しい設定項目 '{key}' をデフォルト値 {default_value} で追加")
    
    # マージした設定を保存
    if app_settings != settings.get('app_settings_f1', {}):
        settings['app_settings_f1'] = app_settings
        save_settings(settings)
    
    return app_settings

def save_app_settings_f1(app_settings):
    """F1のアプリケーション設定を保存"""
    settings = load_settings()
    
    # 保存すべきキーのみを含める（許可リスト方式）
    allowed_keys = [
        'resolution', 'mp4_crf', 'steps', 'cfg', 'use_teacache',
        'gpu_memory_preservation', 'gs', 'image_strength',
        'keep_section_videos', 'save_section_frames', 'save_tensor_data',
        'frame_save_mode', 'save_settings_on_start', 'alarm_on_completion'
    ]
    
    filtered_settings = {k: v for k, v in app_settings.items() 
                        if k in allowed_keys}
    
    settings['app_settings_f1'] = filtered_settings
    return save_settings(settings)

def load_app_settings_oichi():
    """oichiのアプリケーション設定を読み込む"""
    settings = load_settings()
    
    # oichiの設定が存在しない場合はデフォルト値を設定
    if 'app_settings_oichi' not in settings:
        settings['app_settings_oichi'] = get_default_app_settings_oichi()
        save_settings(settings)
    
    # 既存の設定にデフォルト値をマージ（新しいキーのため）
    app_settings = settings.get('app_settings_oichi', {})
    default_settings = get_default_app_settings_oichi()
    
    # 存在しないキーにはデフォルト値を使用
    for key, default_value in default_settings.items():
        if key not in app_settings:
            app_settings[key] = default_value
            print(f"[INFO] oichi: 新しい設定項目 '{key}' をデフォルト値 {default_value} で追加")
    
    # マージした設定を保存
    if app_settings != settings.get('app_settings_oichi', {}):
        settings['app_settings_oichi'] = app_settings
        save_settings(settings)
    
    return app_settings

def save_app_settings_oichi(app_settings):
    """oichiのアプリケーション設定を保存"""
    settings = load_settings()
    
    # 不要なキーを除外してコピー
    filtered_settings = {k: v for k, v in app_settings.items() 
                        if k not in ['rs', 'output_dir']}
    
    settings['app_settings_oichi'] = filtered_settings
    return save_settings(settings)
