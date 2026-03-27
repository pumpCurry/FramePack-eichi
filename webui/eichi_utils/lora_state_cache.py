"""
LoRA ステートキャッシュ

LoRA+FP8 マージ済み state_dict をメモリ/ディスクにキャッシュし、
2回目以降の生成で高価なマージ処理をスキップする。

保存形式は safetensors (推奨) と pt (レガシー) の2形式に対応。
読み込みは両形式をフォールバックで試行する。
"""

import os
import sys
import hashlib
import threading

# ====================================================================
# グローバル設定
# ====================================================================
cache_enabled = False

# 保存形式: "safetensors" or "pt"
_preferred_format = "safetensors"
_SUPPORTED_EXTS = (".safetensors", ".pt")

# ====================================================================
# オンメモリキャッシュ（プロセス内シングルトン）
# ====================================================================
_INMEM_CACHE = {}
_INMEM_LOCK = threading.Lock()


def _inmem_get(cache_key):
    """スレッド安全にオンメモリキャッシュを取得"""
    with _INMEM_LOCK:
        return _INMEM_CACHE.get(cache_key)


def _inmem_set(cache_key, state_dict):
    """スレッド安全にオンメモリキャッシュへ保存"""
    with _INMEM_LOCK:
        _INMEM_CACHE[cache_key] = state_dict


def _inmem_pop(cache_key):
    """特定キーをオンメモリキャッシュから削除"""
    with _INMEM_LOCK:
        return _INMEM_CACHE.pop(cache_key, None)


def _inmem_clear():
    """オンメモリキャッシュを全てクリア"""
    with _INMEM_LOCK:
        _INMEM_CACHE.clear()


# ====================================================================
# キャッシュ有効/無効
# ====================================================================
def set_cache_enabled(value: bool):
    """Toggle persistent LoRA state caching."""
    global cache_enabled
    cache_enabled = bool(value)
    print(f"LoRA state cache enabled: {cache_enabled}")


def is_cache_enabled() -> bool:
    """現在のキャッシュ有効/無効状態を返す"""
    return cache_enabled


# ====================================================================
# 保存形式の切り替え
# ====================================================================
def set_preferred_format(fmt: str):
    """保存形式を設定する ("safetensors" or "pt")"""
    global _preferred_format
    fmt = str(fmt).strip().lower()
    if fmt not in ("safetensors", "pt"):
        fmt = "safetensors"
    _preferred_format = fmt
    print(f"LoRA cache format: {_preferred_format}")


def get_preferred_format() -> str:
    """現在の保存形式を返す"""
    return _preferred_format


# ====================================================================
# model_files 記憶 (peek_next_cache_path 用)
# ====================================================================
_LAST_MODEL_FILES = []
_MODEL_FILES_LOCK = threading.Lock()


def register_last_model_files(model_files):
    """TransformerManager._find_model_files() の結果を記憶する。
    peek_next_cache_path が正確なキャッシュキーを生成するために使用。"""
    global _LAST_MODEL_FILES
    with _MODEL_FILES_LOCK:
        _LAST_MODEL_FILES = list(model_files or [])


def get_last_model_files():
    """記憶済みの model_files を返す"""
    with _MODEL_FILES_LOCK:
        return list(_LAST_MODEL_FILES)


# ====================================================================
# キャッシュディレクトリ / キー生成
# ====================================================================
def get_cache_dir():
    """Return directory for cached LoRA state dictionaries."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cache_dir = os.path.join(base_dir, 'lora_state_cache')
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def generate_cache_key(model_files, lora_paths, lora_scales, fp8_enabled):
    """Generate a unique key from model/LoRA files and settings.
    キーは拡張子に依存しない (ハッシュ文字列のみ)。"""
    items = []

    # model files are order independent
    for path in sorted([str(p) for p in (model_files or [])]):
        if os.path.exists(path):
            items.append(path)
            items.append(str(os.path.getmtime(path)))

    # keep LoRA paths paired with their scale values when sorting
    if lora_paths:
        scales = lora_scales or [None] * len(lora_paths)
        pairs = [(str(p), s) for p, s in zip(lora_paths, scales)]
        for path, scale in sorted(pairs, key=lambda x: x[0]):
            if os.path.exists(path):
                items.append(path)
                items.append(str(os.path.getmtime(path)))
                if scale is not None:
                    items.append(str(scale))

    items.append('fp8' if fp8_enabled else 'no_fp8')
    key_str = '|'.join(items)
    return hashlib.sha256(key_str.encode('utf-8')).hexdigest()


# ====================================================================
# peek: 次のキャッシュファイルパスを予測
# ====================================================================
def peek_next_cache_path(lora_paths=None, lora_scales=None,
                         fp8_enabled=False, force_dict_split=False):
    """次のキャッシュファイルのパスを予測して返す（存在チェック用）。
    RAMガードがキャッシュファイルサイズを事前判定するために使用。
    register_last_model_files() で記憶された model_files を使用する。
    ファイルが存在しない場合は None を返す。"""
    if not cache_enabled:
        return None
    try:
        model_files = get_last_model_files()
        cache_key = generate_cache_key(
            model_files, lora_paths or [], lora_scales or [], fp8_enabled
        )
        cache_dir = get_cache_dir()
        # 両形式をチェック
        for ext in _SUPPORTED_EXTS:
            path = os.path.join(cache_dir, cache_key + ext)
            if os.path.exists(path):
                return path
    except Exception:
        pass
    return None


# ====================================================================
# デュアルフォーマット保存/読み込み (内部)
# ====================================================================
def _save_state_dict(path_without_ext, state_dict):
    """preferred format で state_dict をディスクに保存する。"""
    if _preferred_format == "safetensors":
        try:
            import safetensors.torch as sf
            sf.save_file(state_dict, path_without_ext + ".safetensors")
            return path_without_ext + ".safetensors"
        except Exception as e:
            print(f"safetensors save failed, falling back to pt: {e}")

    # fallback to pt
    import torch
    torch.save(state_dict, path_without_ext + ".pt")
    return path_without_ext + ".pt"


def _load_state_dict(path_without_ext):
    """両形式をフォールバックで試行し、最初に見つかったものを返す。
    見つからない場合は None を返す。"""
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
            # 古い torch: weights_only 未対応
            import torch
            return torch.load(path, map_location="cpu")
        except Exception as e:
            print(f"Cache load failed for {path}: {e}")
            continue
    return None


# ====================================================================
# 公開API: load / save
# ====================================================================
def load_from_cache(cache_key):
    """キャッシュがあれば読み込み、なければ None を返す（オンメモリ優先）"""
    try:
        from webui.locales.i18n_extended import translate
    except Exception:
        translate = lambda x: x  # noqa: E731

    # ① まずオンメモリキャッシュを確認
    mem = _inmem_get(cache_key)
    if mem is not None:
        print(translate("オンメモリのLoRA キャッシュを再利用します: {0}").format(cache_key[:16]))
        return mem

    cache_dir = get_cache_dir()
    path_no_ext = os.path.join(cache_dir, cache_key)

    print(translate("出力済みLoRA キャッシュを読み込んでいます: {0}").format(cache_key[:16]))

    # ② ディスクキャッシュ読み込み (デュアルフォーマット)
    try:
        obj = _load_state_dict(path_no_ext)
        if obj is None:
            print(translate("LoRA キャッシュ Miss"))
            return None

        # ③ 読み込んだデータをオンメモリに保存
        _inmem_set(cache_key, obj)
        print(translate("LoRA キャッシュ Hit"))
        return obj

    except Exception as e:
        print(translate("LoRA キャッシュ が読み込めません: {0}").format(cache_key[:16]))
        print(translate("エラー内容: {0}").format(e))
        print(translate("キャッシュが得られなかったので、最適化処理及び再生成します"))
        return None


def save_to_cache(cache_key, state_dict):
    """現在の LoRA 状態をキャッシュに保存する（オンメモリ＋ディスク）"""
    try:
        from webui.locales.i18n_extended import translate
    except Exception:
        translate = lambda x: x  # noqa: E731

    # ① 先にオンメモリへ登録
    _inmem_set(cache_key, state_dict)

    cache_dir = get_cache_dir()
    os.makedirs(cache_dir, exist_ok=True)
    path_no_ext = os.path.join(cache_dir, cache_key)

    print(translate("メモリ上のLoRA キャッシュを書き出しています: {0}").format(
        cache_key[:16] + "." + _preferred_format))

    try:
        saved_path = _save_state_dict(path_no_ext, state_dict)
        print(translate("メモリ上のLoRA キャッシュの書き出しに成功: {0}").format(
            os.path.basename(saved_path)))
    except Exception as e:
        print(translate("メモリ上のLoRA キャッシュの書き出しに失敗: {0}").format(
            cache_key[:16]))
        print(translate("エラー内容: {0}").format(e))


def _echo_fetching_cache(title: str) -> None:
    """tqdm が使えればミニ進捗（1/1）、無ければ簡易表示"""
    try:
        from tqdm import tqdm
        for _ in tqdm(range(1), desc=title, unit="it"):
            pass
    except Exception:
        print(f"{title} ...")
        sys.stdout.flush()
