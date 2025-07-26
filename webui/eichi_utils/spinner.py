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

    def spinner():
        while not done.is_set():
            sys.stdout.write(f"{next(spinner_cycle)} {message}")
            sys.stdout.flush()
            time.sleep(0.1)
            sys.stdout.write('\r')

    spinner_thread = threading.Thread(target=spinner)
    spinner_thread.start()
    try:
        result = function(*args, **kwargs)
    finally:
        done.set()
        spinner_thread.join()
        sys.stdout.write("✅")
        sys.stdout.write('\n')
    return result
