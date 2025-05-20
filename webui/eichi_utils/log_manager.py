"""
ログ管理モジュール
コンソール出力をファイルにもリダイレクトする機能を提供
"""

import os
import sys
import datetime
from locales.i18n_extended import translate

# グローバル変数
_log_enabled = False
_log_folder = "logs"  # デフォルトのログフォルダ
_log_file = None  # ログファイルのハンドル
_original_stdout = sys.stdout  # 元のstdoutを保存

# webuiディレクトリのパスを取得
_webui_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 絶対パスを取得するヘルパー関数
def get_absolute_path(path):
    """
    相対パスを絶対パスに変換する
    """
    if os.path.isabs(path):
        return path
    else:
        return os.path.normpath(os.path.join(_webui_path, path))

class LoggerWriter:
    """
    標準出力をラップしてログファイルにも書き込むクラス
    """
    def __init__(self, original_stream, log_file):
        self.original_stream = original_stream
        self.log_file = log_file

    def write(self, text):
        # 元のストリームに書き込む
        self.original_stream.write(text)
        
        # ログファイルにも書き込む
        if self.log_file and not self.log_file.closed:
            self.log_file.write(text)
            self.log_file.flush()  # すぐに書き込みを反映
    
    def flush(self):
        self.original_stream.flush()
        if self.log_file and not self.log_file.closed:
            self.log_file.flush()
    
    # その他の必要なメソッド
    def fileno(self):
        return self.original_stream.fileno()
    
    def isatty(self):
        return self.original_stream.isatty()

def enable_logging(log_folder=None, source_name="endframe_ichi"):
    """
    ログ機能を有効化し、標準出力をログファイルにリダイレクトする
    
    Args:
        log_folder: ログフォルダのパス
        source_name: ソースファイル名（拡張子無し）。デフォルトは"endframe_ichi"
    """
    global _log_enabled, _log_folder, _log_file, _original_stdout
    
    if log_folder:
        # 相対パスを絶対パスに変換
        _log_folder = get_absolute_path(log_folder)
    
    # 診断用のメッセージを表示
    print(f"[DEBUG] enable_logging呼び出し: folder={_log_folder}, source={source_name}")
    
    # ログディレクトリが存在しない場合は作成
    try:
        if not os.path.exists(_log_folder):
            os.makedirs(_log_folder, exist_ok=True)
            print(translate("ログフォルダを作成しました: {0}").format(_log_folder))
    except Exception as e:
        print(translate("ログフォルダの作成に失敗しました: {0} - {1}").format(_log_folder, str(e)))
        return False
    
    # 既にログが有効な場合は一度無効化してから再設定
    if _log_enabled:
        disable_logging()
    
    # ソースファイル名からベース名を取得
    # パスが含まれていたり、拡張子が付いていた場合は除去
    base_name = os.path.basename(source_name)
    if base_name.endswith(".py"):
        base_name = base_name[:-3]  # .pyを除去
    
    # 現在の日時をファイル名に使用
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{base_name}_{timestamp}.log"
    log_filepath = os.path.join(_log_folder, log_filename)
    
    try:
        # フルパスを表示
        print(f"[DEBUG] ログファイルパス: {os.path.abspath(log_filepath)}")
        
        # ログファイルを開く
        _log_file = open(log_filepath, "w", encoding="utf-8")
        
        # ログヘッダー情報を書き込む
        header = f"=== FramePack: {base_name} ログ開始 {timestamp} ===\n"
        _log_file.write(header)
        _log_file.flush()
        
        # 標準出力をラッパークラスで置き換え
        _original_stdout = sys.stdout
        sys.stdout = LoggerWriter(_original_stdout, _log_file)
        
        _log_enabled = True
        print(translate("ログ出力を有効化しました: {0}").format(log_filepath))
        return True
    except Exception as e:
        print(translate("ログ出力の設定に失敗しました: {0}").format(str(e)))
        # エラーが発生した場合、元の標準出力に戻す
        sys.stdout = _original_stdout
        return False

def disable_logging():
    """
    ログ機能を無効化し、標準出力を元に戻す
    """
    global _log_enabled, _log_file, _original_stdout
    
    if not _log_enabled:
        return
    
    try:
        # 標準出力を元に戻す
        sys.stdout = _original_stdout
        
        # ログファイルを閉じる
        if _log_file and not _log_file.closed:
            # ログフッター情報を書き込む
            footer = f"\n=== ログ終了 {datetime.datetime.now().strftime('%Y%m%d_%H%M%S')} ===\n"
            _log_file.write(footer)
            _log_file.close()
        
        _log_enabled = False
        print(translate("ログ出力を無効化しました"))
        return True
    except Exception as e:
        print(translate("ログ出力の停止に失敗しました: {0}").format(str(e)))
        return False

def is_logging_enabled():
    """
    ログ機能が有効かどうかを返す
    """
    return _log_enabled

def get_log_folder():
    """
    現在のログフォルダを返す
    """
    return _log_folder

def set_log_folder(folder_path):
    """
    ログフォルダを設定する
    """
    global _log_folder
    
    print(f"[DEBUG] set_log_folder呼び出し: 現在={_log_folder}, 新規={folder_path}")
    
    # 現在ログが有効な場合は一度無効化する
    if _log_enabled:
        disable_logging()
    
    # 相対パスを絶対パスに変換
    _log_folder = get_absolute_path(folder_path)
    
    # フォルダが存在しなければ作成
    if not os.path.exists(_log_folder):
        try:
            os.makedirs(_log_folder, exist_ok=True)
            print(f"[DEBUG] 新しいログフォルダを作成: {_log_folder}")
        except Exception as e:
            print(f"[WARNING] ログフォルダの作成に失敗: {e}")
            pass
    
    print(f"[DEBUG] ログフォルダを設定: {os.path.abspath(_log_folder)}")
    return True

def open_log_folder():
    """
    ログフォルダをOSに依存せず開く
    """
    folder_path = _log_folder
    
    if not os.path.exists(folder_path):
        try:
            os.makedirs(folder_path, exist_ok=True)
        except Exception as e:
            print(translate("ログフォルダの作成に失敗しました: {0} - {1}").format(folder_path, str(e)))
            return False
    
    try:
        if os.name == 'nt':  # Windows
            import subprocess
            subprocess.Popen(['explorer', folder_path])
        elif os.name == 'posix':  # Linux/Mac
            import subprocess
            try:
                subprocess.Popen(['xdg-open', folder_path])
            except:
                subprocess.Popen(['open', folder_path])
        print(translate("ログフォルダを開きました: {0}").format(folder_path))
        return True
    except Exception as e:
        print(translate("ログフォルダを開く際にエラーが発生しました: {0}").format(e))
        return False

def get_default_log_settings():
    """
    デフォルトのログ設定を返す
    """
    # 必ず決まった値を返すようにする
    settings = {
        "log_enabled": False,
        "log_folder": "logs"
    }
    print(f"[DEBUG] デフォルトのログ設定: {settings}")
    return settings

def load_log_settings(app_settings):
    """
    設定からログ設定を読み込む
    """
    if not app_settings:
        return get_default_log_settings()
    
    log_settings = {
        "log_enabled": app_settings.get("log_enabled", False),
        "log_folder": app_settings.get("log_folder", "logs")
    }
    
    return log_settings

def apply_log_settings(log_settings, source_name="endframe_ichi"):
    """
    ログ設定を適用する
    
    Args:
        log_settings: ログ設定辞書
        source_name: ソースファイル名（拡張子無し）。デフォルトは"endframe_ichi"
    """
    if not log_settings:
        print("[WARNING] ログ設定がNoneです")
        return False
    
    print(f"[DEBUG] apply_log_settings: {log_settings}, source={source_name}")
    
    # 現在のログ状態を保存
    was_enabled = is_logging_enabled()
    print(f"[DEBUG] 現在のログ状態: {was_enabled}")
    
    # 一旦ログを無効化（既存のファイルを閉じる）
    if was_enabled:
        print("[DEBUG] 既存のログを一旦無効化します")
        disable_logging()
    
    # ログフォルダを設定
    folder = log_settings.get("log_folder", "logs")
    print(f"[DEBUG] ログフォルダ設定: {folder}")
    set_log_folder(folder)
    
    # ログ有効/無効を設定
    is_enabled = log_settings.get("log_enabled", False)
    print(f"[DEBUG] ログ有効設定: {is_enabled}")
    
    if is_enabled:
        print("[DEBUG] ログを有効化します")
        success = enable_logging(log_folder=folder, source_name=source_name)
        print(f"[DEBUG] ログ有効化結果: {success}")
    
    return True

# プログラム終了時に自動的にログを閉じるようにする
import atexit
atexit.register(disable_logging)