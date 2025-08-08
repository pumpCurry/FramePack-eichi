"""
ログ管理モジュール
コンソール出力をファイルにもリダイレクトする機能を提供
"""

import os
import sys
import datetime
import subprocess
import shutil
from locales.i18n_extended import translate

# グローバル変数
_log_enabled = False
_log_folder = "logs"  # デフォルトのログフォルダ
_log_file = None  # ログファイルのハンドル
_original_stdout = sys.stdout  # 元のstdoutを保存
_original_stderr = sys.stderr  # 元のstderrを保存
_last_progress_percent = None  # 最後に記録した進捗率
_progress_log_interval = 10    # 進捗率の記録間隔（%単位）

# webuiディレクトリのパスを取得
_webui_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 絶対パスを取得するヘルパー関数
def get_absolute_path(path):
    """
    相対パスを絶対パスに変換する

    パスに真偽値やNoneが渡されると ``os.path`` 関数で例外が発生するため、
    その場合は空文字列として扱う。
    """
    if isinstance(path, bool) or path is None:
        path = ""

    if os.path.isabs(path):
        return path
    else:
        return os.path.normpath(os.path.join(_webui_path, str(path)))

class LoggerWriter:
    """
    標準出力をラップしてログファイルにも書き込むクラス
    """
    def __init__(self, original_stream, log_file):
        self.original_stream = original_stream
        self.log_file = log_file

    def write(self, text):
        # 元のストリームに書き込む - コンソールには元のテキストをそのまま表示
        self.original_stream.write(text)
        
        # ログファイルにも書き込む（タイムスタンプ付き）
        if self.log_file and not self.log_file.closed:
            # 空の文字列はスキップ
            if not text.strip():
                return
                
            # プログレスバーチェック - 単純なパターンマッチング
            is_progress_bar = False
            if '\r' in text or '%' in text and ('|' in text or '/' in text):
                is_progress_bar = True
                
            # タイムスタンプを生成
            timestamp = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
            
            if is_progress_bar:
                # プログレスバー処理
                global _last_progress_percent, _progress_log_interval
                
                # テキストからパーセンテージを抽出
                import re
                progress_match = re.search(r'(\d+)%', text)
                if progress_match:
                    percent = int(progress_match.group(1))
                    
                    # 進捗率の間引き処理
                    should_log = (
                        _last_progress_percent is None or  # 最初の進捗
                        percent == 0 or  # 開始
                        percent == 100 or  # 完了
                        _last_progress_percent is not None and 
                        (abs(percent - _last_progress_percent) >= _progress_log_interval)  # 間隔を超えた
                    )
                    
                    if should_log:
                        _last_progress_percent = percent
                        # 追加情報を抽出（例: 25/100）
                        count_match = re.search(r'(\d+)/(\d+)', text)
                        if count_match:
                            current, total = count_match.groups()
                            self.log_file.write(f"{timestamp} [PROGRESS] {percent}% ({current}/{total})\n")
                        else:
                            self.log_file.write(f"{timestamp} [PROGRESS] {percent}%\n")
                else:
                    # パーセンテージが見つからない場合はシンプルにログ
                    progress_text = text.strip().replace('\r', '').split('\n')[-1][:50]
                    self.log_file.write(f"{timestamp} [PROGRESS] {progress_text}\n")
            else:
                # 通常のテキスト - 改行で分割して各行にタイムスタンプを付与
                for line in text.split('\n'):
                    if line.strip():  # 空行をスキップ
                        self.log_file.write(f"{timestamp} {line}\n")
            
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
    global _log_enabled, _log_folder, _log_file, _original_stdout, _original_stderr, _last_progress_percent
    
    if log_folder:
        # 相対パスを絶対パスに変換
        _log_folder = get_absolute_path(log_folder)
    
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
        # ログファイルを開く
        _log_file = open(log_filepath, "w", encoding="utf-8")
        
        # ログヘッダー情報を書き込む
        header = f"=== FramePack: {base_name} ログ開始 {timestamp} ===\n"
        _log_file.write(header)
        _log_file.flush()
        
        # 標準出力をラッパークラスで置き換え
        _original_stdout = sys.stdout
        sys.stdout = LoggerWriter(_original_stdout, _log_file)
        
        # 標準エラー出力もリダイレクト（プログレスバーはstderrに出力されることが多い）
        _original_stderr = sys.stderr
        sys.stderr = LoggerWriter(_original_stderr, _log_file)
        
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
    ログ機能を無効化し、標準出力と標準エラー出力を元に戻す
    """
    global _log_enabled, _log_file, _original_stdout, _original_stderr, _last_progress_percent
    
    if not _log_enabled:
        return
    
    try:
        # 標準出力を元に戻す
        sys.stdout = _original_stdout
        
        # 標準エラー出力を元に戻す
        sys.stderr = _original_stderr
        
        # 進捗情報をリセット
        _last_progress_percent = None
        
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

    if isinstance(folder_path, bool) or folder_path is None:
        # 真偽値やNoneが渡された場合はデフォルトの"logs"を使用
        print(f"[WARNING] 不正なフォルダパスを受け取ったため 'logs' を使用します: {folder_path}")
        folder_path = "logs"
    
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
    
    print(f"[DEBUG] ログフォルダを設定: {os.path.abspath(_log_folder)}")
    return True

def open_log_folder():
    """ログフォルダをOSに依存せず開く"""
    # ブール値が入り込むと os.path.exists で例外となるため正規化
    folder_path = get_absolute_path(_log_folder)
    
    if not os.path.exists(folder_path):
        try:
            os.makedirs(folder_path, exist_ok=True)
        except Exception as e:
            print(translate("ログフォルダの作成に失敗しました: {0} - {1}").format(folder_path, str(e)))
            return False
    
    try:
        if os.name == 'nt':  # Windows
            try:
                os.startfile(folder_path)
            except Exception:
                subprocess.Popen(['explorer', folder_path])
            print(translate("ログフォルダを開きました: {0}").format(folder_path))
        elif os.name == 'posix':
            opener = None
            if sys.platform == 'darwin' and shutil.which('open'):
                opener = 'open'
            else:
                opener = shutil.which('xdg-open') or shutil.which('open')
            if opener:
                subprocess.Popen([opener, folder_path])
                print(translate("ログフォルダを開きました: {0}").format(folder_path))
            else:
                print(translate("xdg-open/open が見つからないため自動でフォルダを開けません: {0}").format(folder_path))
        else:
            print(translate("このOSではフォルダを自動で開く機能はサポートされていません: {0}").format(folder_path))
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