"""
プロンプトエンコーディング キャッシュ

テキストエンコーダ出力 (LLaMA + CLIP テンソル) をディスクに保存し、
同じプロンプトでの再エンコードをスキップする。

保存形式は safetensors (推奨) と pt (レガシー) の2形式に対応。
読み込みは両形式をフォールバックで試行する。
"""

import os
import hashlib

# 保存形式: "safetensors" or "pt"
_preferred_format = "safetensors"
_SUPPORTED_EXTS = (".safetensors", ".pt")


def set_preferred_format(fmt: str):
    """保存形式を設定する ("safetensors" or "pt")"""
    global _preferred_format
    fmt = str(fmt).strip().lower()
    if fmt not in ("safetensors", "pt"):
        fmt = "safetensors"
    _preferred_format = fmt


def get_preferred_format() -> str:
    """現在の保存形式を返す"""
    return _preferred_format


def get_cache_dir():
    """Return the directory for prompt cache files."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cache_dir = os.path.join(base_dir, 'prompt_cache')
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def prompt_hash(prompt: str, n_prompt: str) -> str:
    """Generate an MD5 hash from prompt and negative prompt."""
    combined = f"{prompt or ''}||{n_prompt or ''}"
    return hashlib.md5(combined.encode('utf-8')).hexdigest()


# ====================================================================
# デュアルフォーマット保存/読み込み (内部)
# ====================================================================
def _save_data(path_without_ext, data):
    """preferred format でテンソル dict をディスクに保存する。"""
    if _preferred_format == "safetensors":
        try:
            import safetensors.torch as sf
            sf.save_file(data, path_without_ext + ".safetensors")
            return path_without_ext + ".safetensors"
        except Exception as e:
            print(f"safetensors save failed, falling back to pt: {e}")

    import torch
    torch.save(data, path_without_ext + ".pt")
    return path_without_ext + ".pt"


def _load_data(path_without_ext):
    """両形式をフォールバックで試行し、最初に見つかったものを返す。"""
    # preferred format を先に試す
    order = list(_SUPPORTED_EXTS)
    pref_ext = "." + _preferred_format
    if pref_ext in order:
        order.remove(pref_ext)
        order.insert(0, pref_ext)

    for ext in order:
        path = path_without_ext + ext
        if not os.path.exists(path):
            continue
        try:
            if ext == ".safetensors":
                import safetensors.torch as sf
                return sf.load_file(path, device="cpu")
            else:
                import torch
                return torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            import torch
            return torch.load(path, map_location="cpu")
        except Exception as e:
            print(f"Prompt cache load failed for {path}: {e}")
            continue
    return None


# ====================================================================
# 公開API
# ====================================================================
def load_from_cache(prompt: str, n_prompt: str):
    """Load cached tensors from disk if available (dual format)."""
    cache_hash = prompt_hash(prompt, n_prompt)
    path_no_ext = os.path.join(get_cache_dir(), cache_hash)

    print(f"Looking for prompt cache: {cache_hash[:16]}")

    try:
        data = _load_data(path_no_ext)
        if data is not None:
            print("Prompt cache hit")
            return data
    except Exception:
        print("Failed to load prompt cache")
        return None

    print("Prompt cache miss")
    return None


def save_to_cache(prompt: str, n_prompt: str, data: dict):
    """Save tensors to disk cache (dual format)."""
    cache_hash = prompt_hash(prompt, n_prompt)
    path_no_ext = os.path.join(get_cache_dir(), cache_hash)

    print(f"Saving prompt cache: {cache_hash[:16]}.{_preferred_format}")
    try:
        saved_path = _save_data(path_no_ext, data)
        print(f"Prompt cache saved: {os.path.basename(saved_path)}")
    except Exception as e:
        print(f"Failed to save prompt cache: {e}")
