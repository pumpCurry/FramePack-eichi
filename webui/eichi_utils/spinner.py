import io
import itertools
import sys
import threading
import time
from locales.i18n_extended import translate


def spinner_while_running(message, function, *args, **kwargs):
    """Display a spinner with ``message`` while ``function`` executes."""
    braille_chars = ["⠇", "⠋", "⠙", "⠸", "⢰", "⣠", "⣄", "⡆"]
    spinner_cycle = itertools.cycle(braille_chars)

    done = threading.Event()
    lock = threading.Lock()
    original_stdout = sys.stdout
    buffer = io.StringIO()

    class LockedStdout:
        def write(self, s):
            with lock:
                buffer.write(s)
            return len(s)

        def flush(self):
            pass

        def fileno(self):
            return original_stdout.fileno()

        def isatty(self):
            return original_stdout.isatty()

        @property
        def encoding(self):
            return getattr(original_stdout, "encoding", "utf-8")

    def spinner():
        with lock:
            original_stdout.write(f"{next(spinner_cycle)}  {message}")
            original_stdout.flush()
        while not done.is_set():
            time.sleep(0.1)
            with lock:
                original_stdout.write("\r\x1b[K")
                original_stdout.write(f"{next(spinner_cycle)}  {message}")
                original_stdout.flush()
        with lock:
            original_stdout.write("\r\x1b[K")
            original_stdout.write(f"✅ {message}\n")
            original_stdout.flush()

    spinner_thread = threading.Thread(target=spinner)
    sys.stdout = LockedStdout()
    spinner_thread.start()
    try:
        result = function(*args, **kwargs)
    finally:
        done.set()
        spinner_thread.join()
        sys.stdout = original_stdout
        output = buffer.getvalue()
        if output:
            if not output.startswith("\n"):
                original_stdout.write("\n")
            original_stdout.write(output)
            original_stdout.flush()
    return result
