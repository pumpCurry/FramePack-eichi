"""
キャッシュ統合管理モジュール

LoRAキャッシュ・プロンプトキャッシュの両方を横断的に管理する。
サイズ照会、一括削除、フォーマット情報などの API を提供。
Gradio 非依存 — UI レイヤーは cache_manager_ui.py に分離。
"""

import os

from eichi_utils import lora_state_cache
from eichi_utils import prompt_cache

# 両キャッシュで共通のサポート拡張子
SUPPORTED_EXTS = (".safetensors", ".pt")


# ====================================================================
# ディレクトリ
# ====================================================================
def lora_cache_dir() -> str:
    return lora_state_cache.get_cache_dir()


def prompt_cache_dir() -> str:
    return prompt_cache.get_cache_dir()


# ====================================================================
# エントリ一覧
# ====================================================================
def _scan_cache_entries(cache_dir: str):
    """cache_dir 内の .safetensors / .pt ファイルを列挙する。"""
    entries = []
    try:
        with os.scandir(cache_dir) as it:
            for entry in it:
                if entry.is_file():
                    ext = os.path.splitext(entry.name)[1].lower()
                    if ext in SUPPORTED_EXTS:
                        try:
                            stat = entry.stat()
                            entries.append({
                                "path": entry.path,
                                "name": entry.name,
                                "size_bytes": stat.st_size,
                                "mtime": stat.st_mtime,
                                "format": ext.lstrip("."),
                            })
                        except OSError:
                            continue
    except (FileNotFoundError, PermissionError):
        pass
    return entries


def lora_cache_entries():
    """LoRAキャッシュの全エントリを返す。"""
    return _scan_cache_entries(lora_cache_dir())


def prompt_cache_entries():
    """プロンプトキャッシュの全エントリを返す。"""
    return _scan_cache_entries(prompt_cache_dir())


# ====================================================================
# サイズ集計
# ====================================================================
def lora_cache_total_bytes() -> int:
    return sum(e["size_bytes"] for e in lora_cache_entries())


def prompt_cache_total_bytes() -> int:
    return sum(e["size_bytes"] for e in prompt_cache_entries())


def format_bytes(n: int) -> str:
    """バイト数を人間が読みやすい形式に変換する。"""
    if n < 0:
        return "0 B"
    if n < 1024:
        return f"{n} B"
    elif n < 1024 ** 2:
        return f"{n / 1024:.1f} KB"
    elif n < 1024 ** 3:
        return f"{n / 1024 ** 2:.1f} MB"
    else:
        return f"{n / 1024 ** 3:.2f} GB"


# ====================================================================
# 削除
# ====================================================================
def _clear_cache_dir(cache_dir: str):
    """指定ディレクトリ内のキャッシュファイルを削除する。
    Returns: (deleted_count, freed_bytes)
    Windows でファイルロック中の場合はスキップして続行する。"""
    deleted = 0
    freed = 0
    try:
        with os.scandir(cache_dir) as it:
            for entry in it:
                if entry.is_file():
                    ext = os.path.splitext(entry.name)[1].lower()
                    if ext in SUPPORTED_EXTS:
                        try:
                            size = entry.stat().st_size
                            os.remove(entry.path)
                            deleted += 1
                            freed += size
                        except (PermissionError, OSError):
                            # Windows: ファイルがロック中の場合はスキップ
                            continue
    except (FileNotFoundError, PermissionError):
        pass
    return deleted, freed


def clear_lora_cache(also_clear_inmem: bool = True):
    """LoRAキャッシュを削除する。
    Returns: (deleted_count, freed_bytes)"""
    if also_clear_inmem:
        lora_state_cache._inmem_clear()
    return _clear_cache_dir(lora_cache_dir())


def clear_prompt_cache():
    """プロンプトキャッシュを削除する。
    Returns: (deleted_count, freed_bytes)"""
    return _clear_cache_dir(prompt_cache_dir())


def clear_all_caches():
    """両方のキャッシュを削除する。
    Returns: {"lora": (files, bytes), "prompt": (files, bytes)}"""
    return {
        "lora": clear_lora_cache(also_clear_inmem=True),
        "prompt": clear_prompt_cache(),
    }
