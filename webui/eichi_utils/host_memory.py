"""
クロスプラットフォーム ホストメモリ検出

psutil → /proc/meminfo → None のフォールバックチェーンで
Windows / Linux / macOS いずれでも利用可能RAM/総RAMを取得する。
"""

import os


def _read_proc_meminfo_kb(key: str):
    """Linux: /proc/meminfo から指定キーの値を kB 単位で返す。失敗時 None。"""
    try:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if line.startswith(key + ":"):
                    # e.g. "MemAvailable:   12345678 kB"
                    parts = line.split()
                    return int(parts[1])  # kB
    except Exception:
        pass
    return None


def host_mem_available_gb():
    """利用可能RAMをGB単位で返す。取得不能なら None。

    フォールバックチェーン:
    1. psutil.virtual_memory().available  (Win/Linux/Mac)
    2. /proc/meminfo MemAvailable         (Linux without psutil)
    3. None
    """
    # 1. psutil (最も信頼性が高い)
    try:
        import psutil
        return psutil.virtual_memory().available / (1024 ** 3)
    except Exception:
        pass

    # 2. /proc/meminfo (Linux)
    kb = _read_proc_meminfo_kb("MemAvailable")
    if kb is not None:
        return kb / (1024 ** 2)  # kB → GB

    return None


def host_mem_total_gb():
    """総RAMをGB単位で返す。取得不能なら None。"""
    try:
        import psutil
        return psutil.virtual_memory().total / (1024 ** 3)
    except Exception:
        pass

    kb = _read_proc_meminfo_kb("MemTotal")
    if kb is not None:
        return kb / (1024 ** 2)

    return None


def host_mem_snapshot():
    """利用可能/総RAMのスナップショットを返す。

    Returns:
        {"avail_gb": float|None, "total_gb": float|None}
    """
    return {
        "avail_gb": host_mem_available_gb(),
        "total_gb": host_mem_total_gb(),
    }
