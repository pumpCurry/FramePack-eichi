"""eichi_utils.progress_bar の単体テスト"""

import os
import importlib.util

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
spec = importlib.util.spec_from_file_location(
    "progress_bar", os.path.join(ROOT, "webui", "eichi_utils", "progress_bar.py")
)
progress_bar = importlib.util.module_from_spec(spec)
spec.loader.exec_module(progress_bar)

parse_bar_tag = progress_bar.parse_bar_tag
make_simple_bar = progress_bar.make_simple_bar
make_progress_bar_html2 = progress_bar.make_progress_bar_html2
THEME_COLORS = progress_bar.THEME_COLORS


class TestParseBarTag:
    def test_theme_tag(self):
        theme, fg, bg, spinner, text = parse_bar_tag("[THEME=cyan]Encoding...")
        assert theme == "cyan"
        assert text == "Encoding..."
        assert spinner is None

    def test_theme_with_spinner(self):
        theme, fg, bg, spinner, text = parse_bar_tag("[THEME=green spinner=false]Done")
        assert theme == "green"
        assert spinner is False
        assert text == "Done"

    def test_bar_tag(self):
        theme, fg, bg, spinner, text = parse_bar_tag("[BAR fg=#ff0000 bg=#eee]Custom")
        assert theme is None
        assert fg == "#ff0000"
        assert bg == "#eee"
        assert text == "Custom"

    def test_no_tag(self):
        theme, fg, bg, spinner, text = parse_bar_tag("Plain text")
        assert theme is None
        assert fg is None
        assert text == "Plain text"

    def test_empty(self):
        theme, fg, bg, spinner, text = parse_bar_tag("")
        assert text == ""

    def test_none(self):
        theme, fg, bg, spinner, text = parse_bar_tag(None)
        assert text == ""


class TestMakeSimpleBar:
    def test_basic(self):
        html = make_simple_bar(50, "#4fc3f7", "#eee", "Half done")
        assert "50%" in html
        assert "Half done" in html
        assert "#4fc3f7" in html

    def test_zero(self):
        html = make_simple_bar(0, None, None, "Starting")
        assert "0%" in html

    def test_hundred(self):
        html = make_simple_bar(100, None, None, "Complete")
        assert "100%" in html
        assert "hidden" in html  # spinner hidden at 100%

    def test_no_spinner(self):
        html = make_simple_bar(50, None, None, "No spin", spinner=False)
        assert "loader" not in html

    def test_clamp(self):
        html = make_simple_bar(150, None, None, "Over")
        assert "100%" in html
        html2 = make_simple_bar(-10, None, None, "Under")
        assert "0%" in html2


class TestMakeProgressBarHtml2:
    def test_theme_yellow(self):
        html = make_progress_bar_html2(0, "[THEME=yellow]Loading...")
        assert THEME_COLORS["yellow"] in html
        assert "Loading..." in html

    def test_theme_blue(self):
        html = make_progress_bar_html2(50, "[THEME=blue]Sampling 5/10")
        assert THEME_COLORS["blue"] in html

    def test_no_theme(self):
        html = make_progress_bar_html2(25, "Plain hint")
        assert "25%" in html
        assert "Plain hint" in html

    def test_all_themes(self):
        for name, color in THEME_COLORS.items():
            html = make_progress_bar_html2(50, f"[THEME={name}]Test")
            assert color in html, f"Theme {name} color {color} not found"
