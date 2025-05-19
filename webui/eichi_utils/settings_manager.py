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
            'app_settings_eichi': get_default_app_settings()
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

def get_default_app_settings():
    """アプリケーションのデフォルト設定を返す"""
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
        "frame_save_mode": "保存しない",
        
        # 自動保存設定
        "save_settings_on_start": False,
        
        # アラーム設定
        "alarm_on_completion": True
    }

def load_app_settings():
    """アプリケーション設定を読み込む"""
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
    """アプリケーション設定を保存"""
    settings = load_settings()
    
    # 不要なキーを除外してコピー（手動保存と自動保存の一貫性のため）
    filtered_settings = {k: v for k, v in app_settings.items() 
                        if k not in ['rs', 'output_dir', 'frame_size_radio']}
    
    settings['app_settings_eichi'] = filtered_settings
    return save_settings(settings)
