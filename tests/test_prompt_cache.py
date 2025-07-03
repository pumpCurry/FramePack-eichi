import importlib.util
import sys
import types
import pickle

# create fake torch module
fake_torch = types.ModuleType('torch')

def fake_save(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def fake_load(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

fake_torch.save = fake_save
fake_torch.load = fake_load
sys.modules['torch'] = fake_torch

spec = importlib.util.spec_from_file_location(
    'prompt_cache', 'webui/eichi_utils/prompt_cache.py'
)
prompt_cache = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prompt_cache)

def test_save_and_load(tmp_path, monkeypatch):
    monkeypatch.setattr(prompt_cache, 'get_cache_dir', lambda: tmp_path)
    data = {'x': 1}
    prompt_cache.save_to_cache('p', 'n', data)
    cache_file = tmp_path / (prompt_cache.prompt_hash('p', 'n') + '.pt')
    assert cache_file.exists()
    loaded = prompt_cache.load_from_cache('p', 'n')
    assert loaded == data


def test_load_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(prompt_cache, 'get_cache_dir', lambda: tmp_path)
    assert prompt_cache.load_from_cache('a', 'b') is None
