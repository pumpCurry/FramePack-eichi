import importlib.util
import sys
import types

# create fake torch module so the util can be imported without dependencies
sys.modules['torch'] = types.ModuleType('torch')

spec = importlib.util.spec_from_file_location(
    'lora_state_cache', 'webui/eichi_utils/lora_state_cache.py'
)
lora_state_cache = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lora_state_cache)


def test_cache_key_order_insensitive(tmp_path):
    model = tmp_path / 'model.safetensors'
    model.write_text('m')
    a = tmp_path / 'a.safetensors'
    b = tmp_path / 'b.safetensors'
    a.write_text('a')
    b.write_text('b')

    key1 = lora_state_cache.generate_cache_key([
        str(model)
    ], [str(a), str(b)], [0.5, 1.0], False)
    key2 = lora_state_cache.generate_cache_key([
        str(model)
    ], [str(b), str(a)], [1.0, 0.5], False)

    assert key1 == key2


def test_cache_key_accepts_path_objects(tmp_path):
    model = tmp_path / "model.safetensors"
    model.write_text("m")
    lora = tmp_path / "lora.safetensors"
    lora.write_text("l")

    key = lora_state_cache.generate_cache_key([
        model
    ], [lora], [0.5], True)

    assert isinstance(key, str)
    assert len(key) == 64
