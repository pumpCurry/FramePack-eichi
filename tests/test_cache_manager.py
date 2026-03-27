"""eichi_utils.cache_manager の単体テスト

cache_manager は lora_state_cache / prompt_cache をインポートするため、
テスト環境では直接importできない (torch依存)。
format_bytes と _scan_cache_entries / _clear_cache_dir のみ
独立してテストする。
"""

import os


class TestFormatBytes:
    """format_bytes は外部依存なしのピュア関数なので直接テスト可能"""

    @staticmethod
    def _load_func():
        import importlib.util
        ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        spec = importlib.util.spec_from_file_location(
            "cache_manager_partial",
            os.path.join(ROOT, "webui", "eichi_utils", "cache_manager.py"),
        )
        # cache_manager は import 時に lora_state_cache を読むが、
        # format_bytes と _scan/_clear はピュア関数なので
        # ソースを直接読んで eval する
        path = os.path.join(ROOT, "webui", "eichi_utils", "cache_manager.py")
        with open(path, "r", encoding="utf-8") as f:
            src = f.read()
        ns = {"os": os, "__builtins__": __builtins__}
        # lora_state_cache / prompt_cache のインポートをスキップ
        lines = src.split("\n")
        filtered = []
        for line in lines:
            if "from eichi_utils import" in line:
                filtered.append("# " + line)  # コメントアウト
            else:
                filtered.append(line)
        exec("\n".join(filtered), ns)
        return ns

    def test_zero(self):
        ns = self._load_func()
        assert ns["format_bytes"](0) == "0 B"

    def test_negative(self):
        ns = self._load_func()
        assert ns["format_bytes"](-1) == "0 B"

    def test_bytes(self):
        ns = self._load_func()
        assert ns["format_bytes"](512) == "512 B"

    def test_kilobytes(self):
        ns = self._load_func()
        assert "KB" in ns["format_bytes"](1024)

    def test_megabytes(self):
        ns = self._load_func()
        assert "MB" in ns["format_bytes"](1024 * 1024)

    def test_gigabytes(self):
        ns = self._load_func()
        assert "GB" in ns["format_bytes"](1024 ** 3)

    def test_large_gb(self):
        ns = self._load_func()
        assert ns["format_bytes"](50 * 1024 ** 3) == "50.00 GB"


class TestScanAndClear:
    """_scan_cache_entries と _clear_cache_dir のテスト"""

    @staticmethod
    def _load_funcs():
        ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        path = os.path.join(ROOT, "webui", "eichi_utils", "cache_manager.py")
        with open(path, "r", encoding="utf-8") as f:
            src = f.read()
        ns = {"os": os, "__builtins__": __builtins__}
        lines = src.split("\n")
        filtered = []
        for line in lines:
            if "from eichi_utils import" in line:
                filtered.append("# " + line)
            else:
                filtered.append(line)
        exec("\n".join(filtered), ns)
        return ns

    def test_empty_dir(self, tmp_path):
        ns = self._load_funcs()
        entries = ns["_scan_cache_entries"](str(tmp_path))
        assert entries == []

    def test_nonexistent_dir(self):
        ns = self._load_funcs()
        entries = ns["_scan_cache_entries"]("/nonexistent/path")
        assert entries == []

    def test_finds_pt_files(self, tmp_path):
        ns = self._load_funcs()
        (tmp_path / "test.pt").write_bytes(b"x" * 100)
        entries = ns["_scan_cache_entries"](str(tmp_path))
        assert len(entries) == 1
        assert entries[0]["format"] == "pt"
        assert entries[0]["size_bytes"] == 100

    def test_finds_safetensors_files(self, tmp_path):
        ns = self._load_funcs()
        (tmp_path / "test.safetensors").write_bytes(b"x" * 200)
        entries = ns["_scan_cache_entries"](str(tmp_path))
        assert len(entries) == 1
        assert entries[0]["format"] == "safetensors"

    def test_ignores_other_files(self, tmp_path):
        ns = self._load_funcs()
        (tmp_path / "readme.txt").write_bytes(b"ignore")
        (tmp_path / "data.pt").write_bytes(b"keep")
        entries = ns["_scan_cache_entries"](str(tmp_path))
        assert len(entries) == 1

    def test_clear_files(self, tmp_path):
        ns = self._load_funcs()
        (tmp_path / "a.pt").write_bytes(b"x" * 100)
        (tmp_path / "b.safetensors").write_bytes(b"y" * 200)
        deleted, freed = ns["_clear_cache_dir"](str(tmp_path))
        assert deleted == 2
        assert freed == 300

    def test_clear_preserves_non_cache(self, tmp_path):
        ns = self._load_funcs()
        (tmp_path / "keep.txt").write_bytes(b"keep")
        (tmp_path / "remove.pt").write_bytes(b"remove")
        deleted, freed = ns["_clear_cache_dir"](str(tmp_path))
        assert deleted == 1
        assert (tmp_path / "keep.txt").exists()
