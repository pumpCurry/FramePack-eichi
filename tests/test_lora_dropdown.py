import os
import sys
import importlib.util
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
spec = importlib.util.spec_from_file_location(
    "path_utils", os.path.join(ROOT, "webui", "eichi_utils", "path_utils.py")
)
path_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(path_utils)
safe_path_join = path_utils.safe_path_join

def test_safe_path_join_rejects_invalid(tmp_path):
    base = tmp_path
    for val in [False, True, 0, "0", 0.0, "なし", None]:
        assert safe_path_join(base, val) == base

def test_safe_path_join_accepts_str(tmp_path):
    base = tmp_path
    fname = "sample.safetensors"
    path = safe_path_join(base, fname)
    assert path == os.path.join(base, fname)
