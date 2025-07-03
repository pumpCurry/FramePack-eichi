import os
from pathlib import Path

def safe_path_join(base: os.PathLike, *parts: str) -> Path:
    """Join path components while gracefully handling invalid values.

    Parameters
    ----------
    base : os.PathLike
        Base path to join to.
    *parts : str
        Additional path components. Falsy or known invalid values such as
        ``False`` or ``"なし"`` are ignored. Non-string values are cast to
        ``str``.
    """

    base_path = Path(base)
    for p in parts:
        if p in (None, False, True, 0, "0", 0.0, "", "なし"):
            continue
        base_path = base_path / str(p)
    return base_path
