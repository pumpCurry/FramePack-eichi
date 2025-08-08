import os
import importlib.util
import subprocess

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Load settings_manager module
spec = importlib.util.spec_from_file_location(
    'settings_manager', os.path.join(ROOT, 'webui', 'eichi_utils', 'settings_manager.py')
)
settings_manager = importlib.util.module_from_spec(spec)
spec.loader.exec_module(settings_manager)

# Load log_manager module
spec_log = importlib.util.spec_from_file_location(
    'log_manager', os.path.join(ROOT, 'webui', 'eichi_utils', 'log_manager.py')
)
log_manager = importlib.util.module_from_spec(spec_log)
spec_log.loader.exec_module(log_manager)

def test_open_output_folder_wsl(monkeypatch, tmp_path):
    called = {}

    monkeypatch.setattr(settings_manager, '_is_wsl', lambda: True)
    monkeypatch.setattr(settings_manager.shutil, 'which', lambda cmd: 'explorer.exe' if cmd == 'explorer.exe' else None)

    def mock_check_output(args):
        called['check'] = args
        return b'C:\\temp\\folder\n'

    def mock_popen(args):
        called['popen'] = args

    monkeypatch.setattr(subprocess, 'check_output', mock_check_output)
    monkeypatch.setattr(subprocess, 'Popen', mock_popen)

    settings_manager.open_output_folder(tmp_path)

    assert called['check'] == ['wslpath', '-w', str(tmp_path)]
    assert called['popen'] == ['explorer.exe', 'C:\\temp\\folder']


def test_open_log_folder_wsl(monkeypatch, tmp_path):
    called = {}

    monkeypatch.setattr(log_manager, '_is_wsl', lambda: True)
    monkeypatch.setattr(log_manager, '_log_folder', str(tmp_path))
    monkeypatch.setattr(log_manager.shutil, 'which', lambda cmd: 'explorer.exe' if cmd == 'explorer.exe' else None)

    def mock_check_output(args):
        called['check'] = args
        return b'C:\\temp\\logs\n'

    def mock_popen(args):
        called['popen'] = args

    monkeypatch.setattr(subprocess, 'check_output', mock_check_output)
    monkeypatch.setattr(subprocess, 'Popen', mock_popen)

    log_manager.open_log_folder()

    assert called['check'] == ['wslpath', '-w', str(tmp_path)]
    assert called['popen'] == ['explorer.exe', 'C:\\temp\\logs']
