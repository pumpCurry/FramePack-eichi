"""eichi_utils.script_loader の単体テスト"""

import os
import importlib.util
import tempfile
import shutil

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
spec = importlib.util.spec_from_file_location(
    "script_loader", os.path.join(ROOT, "webui", "eichi_utils", "script_loader.py")
)
script_loader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(script_loader)


class TestGetScriptsDir:
    def test_returns_scripts_dir(self):
        result = script_loader.get_scripts_dir()
        assert result.endswith("scripts")
        assert os.path.isabs(result)


class TestFindJsFiles:
    def test_finds_existing_js_files(self):
        files = script_loader.find_js_files()
        # scripts/ フォルダに modal.js と notification.js がある
        basenames = [os.path.basename(f) for f in files]
        assert "modal.js" in basenames
        assert "notification.js" in basenames

    def test_sorted_order(self):
        files = script_loader.find_js_files()
        basenames = [os.path.basename(f) for f in files]
        assert basenames == sorted(basenames)


class TestBuildHeadScripts:
    def test_generates_script_tags(self):
        html = script_loader.build_head_scripts()
        assert "<script" in html
        assert "modal.js" in html
        assert "notification.js" in html
        assert 'src="/file=' in html

    def test_each_file_gets_own_tag(self):
        html = script_loader.build_head_scripts()
        # modal.js と notification.js で少なくとも2つの<script>タグ
        assert html.count("<script") >= 2
        assert html.count("</script>") >= 2


class TestBuildHeadScriptsInlineFallback:
    def test_generates_inline_script_tags(self):
        html = script_loader.build_head_scripts_inline_fallback()
        assert "<script>" in html
        # インラインなので src= は含まない
        assert 'src=' not in html
        # modal.js の中身の一部が含まれる
        assert "ensureDialog" in html or "modal_dlg" in html


class TestEmptyScriptsDir:
    def test_empty_dir_returns_empty_string(self):
        # 一時ディレクトリでテスト
        tmpdir = tempfile.mkdtemp()
        try:
            # モンキーパッチでscripts_dirを空ディレクトリに
            original = script_loader.get_scripts_dir
            script_loader.get_scripts_dir = lambda: tmpdir
            try:
                html = script_loader.build_head_scripts()
                assert html == ""
            finally:
                script_loader.get_scripts_dir = original
        finally:
            shutil.rmtree(tmpdir)
