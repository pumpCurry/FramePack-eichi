import logging
import traceback

logger = logging.getLogger("eichi")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.FileHandler("eichi.log", encoding="utf-8"))


def log_and_continue(msg: str = "Unhandled exception"):
    """Decorator to log exceptions and continue execution."""

    def _decorator(fn):
        def _wrapper(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except Exception:
                logger.exception(msg)
                print(traceback.format_exc())
                return None
        return _wrapper
    return _decorator
