import os
from pathlib import Path

def safe_path_join(base: os.PathLike, *parts: str) -> Path:
    """Join path components while gracefully handling invalid values.

    Falsy values such as ``False`` or ``"なし"`` are skipped and any non-string
    parts are coerced to ``str`` before joining.
    """

    base_path = Path(base)
    for p in parts:
        if p in (None, False, True, 0, "0", 0.0, "", "なし"):
            continue
        base_path = base_path / str(p)
    return base_path
