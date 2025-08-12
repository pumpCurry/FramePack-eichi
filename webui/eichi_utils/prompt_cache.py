import os
import hashlib
import torch


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


def load_from_cache(prompt: str, n_prompt: str):
    """Load cached tensors from disk if available."""
    cache_file = os.path.join(get_cache_dir(), prompt_hash(prompt, n_prompt) + '.pt')
    print(f"Looking for prompt cache: {cache_file}")
    if os.path.exists(cache_file):
        try:
            data = torch.load(cache_file)
            print("Prompt cache hit")
            return data
        except Exception:
            print("Failed to load prompt cache")
            return None
    print("Prompt cache miss")
    return None


def save_to_cache(prompt: str, n_prompt: str, data: dict):
    """Save tensors to disk cache."""
    cache_file = os.path.join(get_cache_dir(), prompt_hash(prompt, n_prompt) + '.pt')
    print(f"Saving prompt cache: {cache_file}")
    try:
        torch.save(data, cache_file)
    except Exception as e:
        print(f"Failed to save prompt cache: {e}")
