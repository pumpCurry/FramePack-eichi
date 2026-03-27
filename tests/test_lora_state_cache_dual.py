"""lora_state_cache デュアルフォーマット・model_files記憶のテスト"""

import os
import importlib.util

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
spec = importlib.util.spec_from_file_location(
    "lora_state_cache",
    os.path.join(ROOT, "webui", "eichi_utils", "lora_state_cache.py"),
)
lsc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lsc)


class TestModelFilesRegistry:
    def test_initial_empty(self):
        assert lsc.get_last_model_files() == []

    def test_register_and_get(self):
        lsc.register_last_model_files(["/path/a.safetensors", "/path/b.safetensors"])
        result = lsc.get_last_model_files()
        assert len(result) == 2
        assert "/path/a.safetensors" in result

    def test_register_none(self):
        lsc.register_last_model_files(None)
        assert lsc.get_last_model_files() == []

    def test_returns_copy(self):
        lsc.register_last_model_files(["/path/x.pt"])
        a = lsc.get_last_model_files()
        b = lsc.get_last_model_files()
        assert a == b
        assert a is not b  # different list objects


class TestPreferredFormat:
    def test_default_safetensors(self):
        assert lsc.get_preferred_format() == "safetensors"

    def test_set_pt(self):
        lsc.set_preferred_format("pt")
        assert lsc.get_preferred_format() == "pt"
        lsc.set_preferred_format("safetensors")  # restore

    def test_set_invalid_falls_back(self):
        lsc.set_preferred_format("invalid")
        assert lsc.get_preferred_format() == "safetensors"

    def test_set_none_falls_back(self):
        lsc.set_preferred_format(None)
        assert lsc.get_preferred_format() == "safetensors"


class TestCacheEnabled:
    def test_default_disabled(self):
        assert lsc.is_cache_enabled() is False

    def test_toggle(self):
        lsc.set_cache_enabled(True)
        assert lsc.is_cache_enabled() is True
        lsc.set_cache_enabled(False)
        assert lsc.is_cache_enabled() is False


class TestInmemCache:
    def test_set_get(self):
        lsc._inmem_set("test_key", {"data": 123})
        result = lsc._inmem_get("test_key")
        assert result == {"data": 123}

    def test_get_missing(self):
        assert lsc._inmem_get("nonexistent_key_xyz") is None

    def test_clear(self):
        lsc._inmem_set("clear_test", {"x": 1})
        lsc._inmem_clear()
        assert lsc._inmem_get("clear_test") is None

    def test_pop(self):
        lsc._inmem_set("pop_test", {"y": 2})
        result = lsc._inmem_pop("pop_test")
        assert result == {"y": 2}
        assert lsc._inmem_get("pop_test") is None


class TestGenerateCacheKey:
    def test_deterministic(self):
        k1 = lsc.generate_cache_key([], [], [], False)
        k2 = lsc.generate_cache_key([], [], [], False)
        assert k1 == k2

    def test_fp8_differs(self):
        k1 = lsc.generate_cache_key([], [], [], False)
        k2 = lsc.generate_cache_key([], [], [], True)
        assert k1 != k2

    def test_sha256_length(self):
        k = lsc.generate_cache_key([], [], [], False)
        assert len(k) == 64  # SHA-256 hex
