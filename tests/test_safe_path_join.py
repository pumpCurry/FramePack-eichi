import os
import importlib.util
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
spec = importlib.util.spec_from_file_location(
    'path_utils', os.path.join(ROOT, 'webui', 'eichi_utils', 'path_utils.py')
)
path_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(path_utils)

safe_path_join = path_utils.safe_path_join

@pytest.mark.parametrize('val', [False, True, 0, '', 'なし', None])
def test_skip(val, tmp_path):
    assert safe_path_join(tmp_path, val) == tmp_path


def test_accept(tmp_path):
    fname = 'a.safetensors'
    assert safe_path_join(tmp_path, fname) == tmp_path / fname
