"""prompt_cache デュアルフォーマットのテスト"""

import os
import importlib.util

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
spec = importlib.util.spec_from_file_location(
    "prompt_cache",
    os.path.join(ROOT, "webui", "eichi_utils", "prompt_cache.py"),
)
pc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pc)


class TestPreferredFormat:
    def test_default_safetensors(self):
        assert pc.get_preferred_format() == "safetensors"

    def test_set_pt(self):
        pc.set_preferred_format("pt")
        assert pc.get_preferred_format() == "pt"
        pc.set_preferred_format("safetensors")

    def test_set_invalid(self):
        pc.set_preferred_format("xyz")
        assert pc.get_preferred_format() == "safetensors"


class TestPromptHash:
    def test_deterministic(self):
        h1 = pc.prompt_hash("hello", "world")
        h2 = pc.prompt_hash("hello", "world")
        assert h1 == h2

    def test_different_prompts(self):
        h1 = pc.prompt_hash("a", "b")
        h2 = pc.prompt_hash("c", "d")
        assert h1 != h2

    def test_md5_length(self):
        h = pc.prompt_hash("test", "")
        assert len(h) == 32  # MD5 hex

    def test_empty_prompts(self):
        h = pc.prompt_hash("", "")
        assert isinstance(h, str)
        assert len(h) == 32

    def test_none_prompts(self):
        h = pc.prompt_hash(None, None)
        assert isinstance(h, str)


class TestCacheDir:
    def test_returns_string(self):
        d = pc.get_cache_dir()
        assert isinstance(d, str)
        assert "prompt_cache" in d

    def test_dir_exists(self):
        d = pc.get_cache_dir()
        assert os.path.isdir(d)
