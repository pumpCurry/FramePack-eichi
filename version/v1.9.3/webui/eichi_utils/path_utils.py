import os

def safe_path_join(base, *parts):
    """Join path components only for valid non-empty strings."""
    for p in parts:
        if isinstance(p, str) and p and p not in {"0", "なし"}:
            base = os.path.join(base, p)
    return base
