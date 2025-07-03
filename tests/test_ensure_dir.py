import pathlib, importlib.util, pytest
spec = importlib.util.spec_from_file_location(
    "pu", "webui/eichi_utils/path_utils.py"
)
pu = importlib.util.module_from_spec(spec); spec.loader.exec_module(pu)

@pytest.mark.parametrize("val", [False, True, None, "", "なし"])
def test_default(val):
    assert pu.ensure_dir(val, "out").name == "out"

def test_custom(tmp_path):
    p = tmp_path / "xyz"
    assert pu.ensure_dir(str(p)) == p
