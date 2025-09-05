import importlib.util
import json
import sys
import types

# fake locales module for translation
locales = types.ModuleType('locales')
locales.i18n_extended = types.ModuleType('locales.i18n_extended')
locales.i18n_extended.translate = lambda x: x
sys.modules['locales'] = locales
sys.modules['locales.i18n_extended'] = locales.i18n_extended

spec = importlib.util.spec_from_file_location(
    'settings_manager', 'webui/eichi_utils/settings_manager.py')
sm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sm)


def test_save_app_settings_eichi(tmp_path, monkeypatch):
    settings_file = tmp_path / 'app_settings.json'
    monkeypatch.setattr(sm, 'get_settings_file_path', lambda: settings_file)
    sm.initialize_settings()

    sm.save_app_settings({'resolution': 640, 'lora_cache': True})

    data = json.loads(settings_file.read_text())
    assert data['app_settings_eichi']['lora_cache'] is True


def test_save_app_settings_f1(tmp_path, monkeypatch):
    settings_file = tmp_path / 'app_settings.json'
    monkeypatch.setattr(sm, 'get_settings_file_path', lambda: settings_file)
    sm.initialize_settings()

    sm.save_app_settings_f1({'resolution': 640, 'lora_cache': True})

    data = json.loads(settings_file.read_text())
    assert data['app_settings_f1']['lora_cache'] is True

