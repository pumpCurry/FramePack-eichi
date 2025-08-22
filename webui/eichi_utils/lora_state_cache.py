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
    """Load cached state dict if available."""
    cache_file = os.path.join(get_cache_dir(), cache_key + '.pt')
    print(f"Looking for LoRA state cache: {cache_file}")
    if os.path.exists(cache_file):
        try:
            data = torch.load(cache_file)
            print("LoRA state cache hit")
            return data
        except Exception as e:
            print(f"Failed to load LoRA state cache: {e}")
            return None
    print("LoRA state cache miss")
    return None


def save_to_cache(cache_key, state_dict):
    """Save state dict to cache."""
    cache_file = os.path.join(get_cache_dir(), cache_key + '.pt')
    print(f"Saving LoRA state cache: {cache_file}")
    try:
        torch.save(state_dict, cache_file)
    except Exception as e:
        print(f"Failed to save LoRA state cache: {e}")
