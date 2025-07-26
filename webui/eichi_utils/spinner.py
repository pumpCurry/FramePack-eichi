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

    class LockedStdout:
        def write(self, s):
            with lock:
                original_stdout.write(s)
                original_stdout.flush()

        def flush(self):
            with lock:
                original_stdout.flush()

    def spinner():
        with lock:
            original_stdout.write(f"{next(spinner_cycle)} {message}\n")
            original_stdout.flush()
        while not done.is_set():
            time.sleep(0.1)
            with lock:
                original_stdout.write("\x1b[1A\r\x1b[2K")
                original_stdout.write(f"{next(spinner_cycle)} {message}\n")
                original_stdout.flush()
        with lock:
            original_stdout.write("\x1b[1A\r\x1b[2K")
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
    return result
