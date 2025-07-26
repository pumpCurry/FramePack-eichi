import itertools
import sys
import threading
import time
from locales.i18n_extended import translate
import queue
import contextlib
import io

def spinner_while_running(message, function, *args, **kwargs):
    """Display a spinner with ``message`` while ``function`` executes."""
    braille_chars = ["⠇", "⠋", "⠙", "⠸", "⢰", "⣠", "⣄", "⡆"]
    spinner_cycle = itertools.cycle(braille_chars)

    done = threading.Event()
    print_lock = threading.Lock()
    out_queue = queue.Queue()

    class QueueWriter(io.StringIO):
        def write(self, s):
            if s:
                out_queue.put(s)
        def flush(self):
            pass

    def printer():
        while True:
            chunk = out_queue.get()
            if chunk is None:
                break
            with print_lock:
                sys.stdout.write(chunk)
                sys.stdout.flush()

    def spinner():
        with print_lock:
            sys.stdout.write(f"{next(spinner_cycle)} {message}\n")
            sys.stdout.flush()
        while not done.is_set():
            with print_lock:
                sys.stdout.write('\x1b[1A\r\x1b[2K')
                sys.stdout.write(f"{next(spinner_cycle)} {message}\n")
                sys.stdout.flush()
            time.sleep(0.1)

    result_container = {}

    def run_func():
        with contextlib.redirect_stdout(QueueWriter()):
            result_container['result'] = function(*args, **kwargs)
        out_queue.put(None)

    spinner_thread = threading.Thread(target=spinner)
    printer_thread = threading.Thread(target=printer)
    runner_thread = threading.Thread(target=run_func)

    spinner_thread.start()
    printer_thread.start()
    runner_thread.start()

    runner_thread.join()
    done.set()
    spinner_thread.join()
    printer_thread.join()

    with print_lock:
        sys.stdout.write('\x1b[1A\r\x1b[2K')
        sys.stdout.write(f"✅ {message}\n")
        sys.stdout.flush()

    return result_container.get('result')
