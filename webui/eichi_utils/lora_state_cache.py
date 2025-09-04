import os
import hashlib
import torch

# グローバルキャッシュ設定
cache_enabled = False


def set_cache_enabled(value: bool):
    """Toggle persistent LoRA state caching."""
    global cache_enabled
    cache_enabled = bool(value)
    print(f"LoRA state cache enabled: {cache_enabled}")


def get_cache_dir():
    """Return directory for cached LoRA state dictionaries."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cache_dir = os.path.join(base_dir, 'lora_state_cache')
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def generate_cache_key(model_files, lora_paths, lora_scales, fp8_enabled):
    """Generate a unique key from model/LoRA files and settings."""
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

def load_from_cache(cache_key):
    """キャッシュがあれば読み込み、なければ None を返す（tqdm/スピナー対応）"""
    from locales.i18n_extended import translate
    # from eichi_utils.console import echo_fetching_cache  # tqdm が無ければスピナーにフォールバック

    # 読み込み開始メッセージ
    cache_file = os.path.join(get_cache_dir(), cache_key + '.pt')
    print(translate("出力済みLoRA キャッシュを読み込んでいます: {0}").format(cache_file))

    if os.path.exists(cache_file):
        try:
            # 進捗（tqdm 1/1 即完了 or スピナー）
            _echo_fetching_cache(translate("キャッシュ読み込み中..."))
            data = torch.load(cache_file)
            print(translate("LoRA キャッシュ Hit"))
            return data
        except Exception as e:
            print(translate("LoRA キャッシュ が読み込めません: {0}").format(cache_file))
            print(translate("エラー内容: {0}").format(e))
            print(translate("キャッシュが得られなかったので、最適化処理及び再生成します"))
            return None

    print(translate("LoRA キャッシュ Miss"))
    print(translate("キャッシュがみつからないか、初めて生成します: {0}").format(cache_file))
    return None

def load_from_cache(cache_key):
    """キャッシュがあれば読み込み、なければ None を返す（tqdmで実測％を表示）"""
    import os, torch
    from webui.locales.i18n_extended import translate
    cache_file = os.path.join(get_cache_dir(), cache_key + '.pt')

    print(translate("出力済みLoRA キャッシュを読み込んでいます: {0}").format(cache_key + '.pt'))

    if not os.path.exists(cache_file):
        # キャッシュ無し（ミス）
        print(translate("LoRA キャッシュ Miss"))
        print(translate("キャッシュがみつからないか、初めて生成します: {0}").format(cache_key + '.pt'))
        return None

    # ここからキャッシュあり（ヒット候補）
    try:
        size = os.path.getsize(cache_file)
        # tqdmが利用可能なら実測％、ダメならスピナー
        try:
            from tqdm import tqdm
            with open(cache_file, "rb") as f:
                with tqdm.wrapattr(
                    f, "read",
                    total=size,
                    unit="B", unit_scale=True, unit_divisor=1024,
                    desc=translate("キャッシュ読み込み中")
                ) as wrapped:
                    obj = torch.load(wrapped, map_location=None)
        except Exception:
            # tqdmが使えない／失敗 → 既存スピナー（または簡易出力）でフォールバック
            try:
                # from webui.eichi_utils.console import echo_fetching_cache
                _echo_fetching_cache(translate("キャッシュ読み込み中"))
            except Exception:
                pass
            obj = torch.load(cache_file, map_location=None)

        print(translate("LoRA キャッシュ Hit"))
        return obj

    except Exception as e:
        # 読み込み失敗 → 通常ルートへフォールバック
        print(translate("LoRA キャッシュ が読み込めません: {0}").format(cache_file))
        print(translate("エラー内容: {0}").format(e))
        print(translate("キャッシュが得られなかったので、最適化処理及び再生成します"))
        return None


def save_to_cache(cache_key, state_dict):
    """現在の LoRA 状態をキャッシュに保存する（tqdmで書き込みバイト数を可視化）"""
    import os, torch
    from webui.locales.i18n_extended import translate

    cache_dir = get_cache_dir()
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, cache_key + '.pt')

    print(translate("メモリ上のLoRA キャッシュを書き出しています: {0}").format(cache_file))

    try:
        # tqdmが利用可能なら write をラップ（総サイズは未知のため％は原理的に不可、バイト進捗のみ）
        try:
            from tqdm import tqdm
            with open(cache_file, "wb") as f:
                with tqdm.wrapattr(
                    f, "write",
                    unit="B", unit_scale=True, unit_divisor=1024,
                    desc=translate("キャッシュ書き出し中")
                ) as wrapped:
                    torch.save(state_dict, wrapped)
        except Exception:
            # tqdmが使えない／失敗 → そのまま保存
            with open(cache_file, "wb") as f:
                torch.save(state_dict, f)

        print(translate("メモリ上のLoRA キャッシュの書き出しに成功: {0}").format(cache_file))

    except Exception as e:
        print(translate("メモリ上のLoRA キャッシュの書き出しに失敗: {0}").format(cache_file))
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

