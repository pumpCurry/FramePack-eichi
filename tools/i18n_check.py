# tools/i18n_check.py
# -*- coding: utf-8 -*-
"""
FramePack-eichi i18n Consistency Checker (extended)
- 対象:
  * webui/ および配下（webui/*/…）
  * リポジトリ直下の endframe*.py / endframe_*/*.py / oneframe_oichi*.py
- 仕様:
  * コード（.py/.js/.ts/.tsx/.jsx/.html）から翻訳キーを静的抽出
  * webui/locales/*.json を「その場にある全言語」ぶん自動検出
  * 2系統で過不足を検出:
      (A) 使われているキー全集 vs 各言語
      (B) ベース言語(既定: ja) vs 各言語（将来言語の不足も検出）
  * --fix で JSON を「コード出現順」に並べ替え、欠損はベース言語から補完
  * JSON はコメント不可（RFC 8259）。未使用キーは "_unused" に退避（コメント代替）
  * 走査は .git を確実に除外（dirs[:] の書き換え）
動作要件: Python 3.8+（外部依存なし）
"""

from __future__ import annotations
import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple, Iterable, Optional
from collections import OrderedDict

# ========= 抽出対象の既定 =========
DEFAULT_LOCALES_DIR = "webui/locales"
DEFAULT_INCLUDE_EXTS = {".py", ".js", ".ts", ".tsx", ".jsx", ".html"}

# ディレクトリ／パターン指定:
#  - フォルダは再帰的に走査（webui/）
#  - グロブはリポジトリ直下に対して評価（endframe*.py / oneframe_oichi*.py 等）
DEFAULT_DIR_TARGETS = ["webui"]
DEFAULT_GLOB_TARGETS = [
    "endframe*.py",            # 直下の endframe*.py
    "endframe_*/*.py",         # endframe_*/ を切っている場合
    "oneframe_oichi*.py",      # 直下 oneframe_oichi*.py
]

# .git は必ず除外（ユーザー追加の除外は --exclude-dir で）
DEFAULT_EXCLUDE_DIRS = {".git"}

# 翻訳キー抽出の正規表現（必要に応じて追加）
KEY_PATTERNS = [
    # Python: t("key"), tr('key'), translate("key"), _("key")
    r"""(?<!\w)(?:t|tr|translate|_|i18n\.t)\(\s*["']([^"']+)["']\s*[),]""",
    # JS/TS: i18n.t("key") / t('key')
    r"""i18n\.t\(\s*["']([^"']+)["']\s*\)""",
    r"""(?<!\w)t\(\s*["']([^"']+)["']\s*\)""",
    # テンプレート: {{ t("key") }}
    r"""{{\s*t\(\s*["']([^"']+)["']\s*\)\s*}}""",
]

# ======== JSONC (コメント) を許容したい場合の剥がし処理 ========
_COMMENT_LINE = re.compile(r"^\s*//.*$")
_COMMENT_BLOCK = re.compile(r"/\*.*?\*/", re.DOTALL)

def _strip_jsonc(text: str) -> str:
    text = _COMMENT_BLOCK.sub("", text)
    return "\n".join(line for line in text.splitlines() if not _COMMENT_LINE.match(line))

# ======== ファイル走査 ========
def _iter_files_dir(root: Path,
                    include_exts: Set[str],
                    exclude_dirs: Set[str]) -> Iterable[Path]:
    """ディレクトリを再帰走査し、`.git` などを確実に除外。"""
    for cur_root, dirs, files in os.walk(root, topdown=True):
        # ディレクトリ名ベースで除外（.git など）。ここで枝刈りするのが最も確実。:contentReference[oaicite:4]{index=4}
        dirs[:] = [d for d in dirs if d not in exclude_dirs and d != ".git"]
        for fn in files:
            # `.git` という名前の**ファイル**も除外
            if fn == ".git":
                continue
            p = Path(cur_root) / fn
            if p.suffix.lower() in include_exts:
                yield p

def _iter_files_glob(repo_root: Path, patterns: List[str]) -> Iterable[Path]:
    """リポジトリ直下でグロブ評価（endframe*.py 等）。"""
    for pat in patterns:
        for p in repo_root.glob(pat):
            if p.is_file():
                yield p

def collect_code_files(repo_root: Path,
                       dir_targets: List[str],
                       glob_targets: List[str],
                       include_exts: Set[str],
                       exclude_dirs: Set[str]) -> List[Path]:
    files: Set[Path] = set()
    # ディレクトリ目標
    for d in dir_targets:
        base = (repo_root / d)
        if base.is_dir():
            files.update(_iter_files_dir(base, include_exts, exclude_dirs))
    # グロブ目標
    files.update(_iter_files_glob(repo_root, glob_targets))
    # 表示順の安定化
    return sorted(files, key=lambda p: str(p).replace("\\", "/"))

# ======== キー抽出 ========
def extract_keys_from_text(text: str) -> List[str]:
    keys: List[str] = []
    for pat in KEY_PATTERNS:
        for m in re.finditer(pat, text):
            k = (m.group(1) or "").strip()
            if k:
                keys.append(k)
    return keys

def collect_used_keys(files: List[Path]) -> List[str]:
    seen: Set[str] = set()
    order: List[str] = []
    for path in files:
        try:
            txt = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for k in extract_keys_from_text(txt):
            if k not in seen:
                seen.add(k)
                order.append(k)
    return order

# ======== locales 読み書き ========
def read_json_map(path: Path) -> Dict[str, object]:
    raw = path.read_text(encoding="utf-8")
    try:
        return json.loads(_strip_jsonc(raw) or "{}")
    except Exception as e:
        raise ValueError(f"JSON parse error at {path}: {e}")

def write_json_ordered(path: Path, mapping: OrderedDict) -> None:
    txt = json.dumps(mapping, ensure_ascii=False, indent=2)
    if not txt.endswith("\n"):
        txt += "\n"
    path.write_text(txt, encoding="utf-8")

def find_locale_files(locales_dir: Path) -> Dict[str, Path]:
    """webui/locales/*.json を全列挙し、言語コード→ファイルを返す（将来追加にも対応）"""
    result: Dict[str, Path] = {}
    for p in sorted(locales_dir.glob("*.json")):
        # 例: ru.json -> lang="ru"
        lang = p.stem.strip()
        if lang:
            result[lang] = p
    return result

# ======== 並べ替え・退避 ========
def reorder_locale_by(order_keys: List[str], locale_map: Dict[str, object]) -> OrderedDict:
    od = OrderedDict()
    # 1) 使用キーをコード出現順で
    for k in order_keys:
        if k in locale_map:
            od[k] = locale_map[k]
    # 2) 未使用は末尾の "_unused" に退避
    unused = OrderedDict()
    for k, v in locale_map.items():
        if k not in od and k != "_unused":
            unused[k] = v
    if unused:
        od["_unused"] = unused
    return od

# ======== レポート整形 ========
def print_section(title: str) -> None:
    print("\n" + title)
    print("-" * len(title))

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="repo root")
    ap.add_argument("--locales", default=DEFAULT_LOCALES_DIR, help="locales dir (e.g. webui/locales)")
    ap.add_argument("--base", default="ja", help="base language code (default: ja)")
    ap.add_argument("--fix", action="store_true", help="apply ordering & fill missing from base")
    ap.add_argument("--exclude-dir", action="append", default=[], help="extra directories to exclude (can repeat)")
    ap.add_argument("--ext", action="append", default=[], help="extra file extensions to include (e.g. .vue)")
    ap.add_argument("--no-dir-target", action="store_true", help="ignore default dir targets (webui)")
    ap.add_argument("--no-glob-target", action="store_true", help="ignore default glob targets (endframe*, oneframe_oichi*)")
    args = ap.parse_args()

    repo_root = Path(args.root).resolve()
    locales_dir = (repo_root / args.locales)
    if not locales_dir.exists():
        print(f"[ERROR] locales dir not found: {locales_dir}", file=sys.stderr)
        sys.exit(2)

    include_exts = set(DEFAULT_INCLUDE_EXTS)
    for e in args.ext:
        if not e.startswith("."):
            e = "." + e
        include_exts.add(e.lower())

    exclude_dirs = set(DEFAULT_EXCLUDE_DIRS)
    for d in args.exclude_dir:
        exclude_dirs.add(d)

    dir_targets = [] if args.no_dir_target else list(DEFAULT_DIR_TARGETS)
    glob_targets = [] if args.no_glob_target else list(DEFAULT_GLOB_TARGETS)

    # ---- 1) コード走査 → 使用キー順を確定 ----
    code_files = collect_code_files(repo_root, dir_targets, glob_targets, include_exts, exclude_dirs)
    used_order = collect_used_keys(code_files)
    used_set = set(used_order)

    print_section("Scan Summary")
    print(f"Code files scanned : {len(code_files)}")
    print(f"Used keys detected : {len(used_order)}")

    # ---- 2) ロケール列挙（将来追加にも自動対応）----
    locale_files = find_locale_files(locales_dir)
    if not locale_files:
        print(f"[ERROR] no locale files (*.json) under {locales_dir}", file=sys.stderr)
        sys.exit(2)

    print(f"Locales detected   : {', '.join(sorted(locale_files.keys()))}")

    # ---- 3) 各ロケール読み込み ----
    locales: Dict[str, Dict[str, object]] = {}
    for lang, p in locale_files.items():
        try:
            mp = read_json_map(p)
            if not isinstance(mp, dict):
                raise ValueError("top-level JSON must be an object")
            locales[lang] = mp
        except Exception as e:
            print(f"[ERROR] {lang}: {e}", file=sys.stderr)
            sys.exit(2)

    base_lang = args.base if args.base in locales else sorted(locale_files.keys())[0]
    base_map = locales.get(base_lang, {})

    # ---- 4) (A) 使われているキー vs 各言語 ／ (B) ベース vs 各言語 ----
    overall_failed = False

    print_section("Per-Locale Report")
    for lang in sorted(locales.keys()):
        lm = locales[lang]
        lm_keys = [k for k in lm.keys() if k != "_unused"]
        lm_key_set = set(lm_keys)

        # (A) 使用キーの不足・余剰
        missing_used = [k for k in used_order if k not in lm]
        extra_used = sorted(lm_key_set - used_set)

        # (B) ベース言語との過不足（将来言語の整備状況も把握）
        base_only = sorted(set(base_map.keys()) - lm_key_set - {"_unused"})
        lang_only = sorted(lm_key_set - set(base_map.keys()) - {"_unused"})

        # 簡易の順序ズレ判定（先頭20件比較）
        file_order_head = [k for k in lm_keys][:20]
        should_head = [k for k in used_order if k in lm][:20]
        order_ok = (file_order_head == should_head)

        print(f"[{lang}]")
        print(f"  missing (used)   : {len(missing_used)}")
        if missing_used:
            for k in missing_used[:50]:
                print(f"    - {k}")
            if len(missing_used) > 50:
                print(f"    ... and {len(missing_used) - 50} more")
        print(f"  extra   (used)   : {len(extra_used)}")
        if extra_used:
            for k in extra_used[:50]:
                print(f"    - {k}")
            if len(extra_used) > 50:
                print(f"    ... and {len(extra_used) - 50} more")

        print(f"  missing vs {base_lang}: {len(base_only)}")
        if base_only:
            for k in base_only[:50]:
                print(f"    - {k}")
            if len(base_only) > 50:
                print(f"    ... and {len(base_only) - 50} more")
        print(f"  extra   vs {base_lang}: {len(lang_only)}")
        if lang_only:
            for k in lang_only[:50]:
                print(f"    - {k}")
            if len(lang_only) > 50:
                print(f"    ... and {len(lang_only) - 50} more")

        print(f"  order_ok         : {order_ok}")

        if missing_used or (not order_ok):
            overall_failed = True

        # --fix: 欠損補完 + コード順に再整列（未使用は _unused へ）
        if args.fix:
            fixed = dict(lm)  # shallow copy
            # 使われているのに無いキーは base から補完（無ければキー名）
            for k in missing_used:
                fixed[k] = base_map.get(k, k)
            # コード順に並び替え
            ordered = reorder_locale_by(used_order, fixed)
            write_json_ordered(locale_files[lang], ordered)
            print(f"  [FIXED] wrote: {locale_files[lang]}")

    print_section("Result")
    if overall_failed:
        print("❌ Inconsistencies found (missing keys and/or order mismatch). Run with --fix to repair.")
        sys.exit(1)
    else:
        print("✅ All locales consistent with code usage and ordering.")
        sys.exit(0)

if __name__ == "__main__":
    main()
