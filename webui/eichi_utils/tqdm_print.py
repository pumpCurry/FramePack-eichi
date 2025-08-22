import builtins
import sys
from tqdm import tqdm

_original_print = builtins.print

def tqdm_print(*args, **kwargs):
    """Thread-safe print function that uses tqdm.write to avoid breaking progress displays."""
    file = kwargs.get("file", None)
    if file not in (None, sys.stdout):
        _original_print(*args, **kwargs)
        return
    sep = kwargs.get("sep", " ")
    end = kwargs.get("end", "\n")
    msg = sep.join(str(a) for a in args)
    tqdm.write(msg, end=end)


def enable_tqdm_print():
    """Replace the built-in print with a tqdm-based version."""
    builtins.print = tqdm_print
