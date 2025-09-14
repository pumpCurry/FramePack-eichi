# tools/i18n_check.py
# -*- coding: utf-8 -*-
"""
FramePack-eichi i18n Consistency Checker (safe, 2-phase)
- まず全コードを走査して使用キー順を一意に確定
- 各ロケールの「期待結果」をメモリ上で作成 → 現状と厳密比較（キー集合＋順序＋値）
- 差分があるファイルだけを最後にまとめて原子的に置換
- 既定ではキー削除しない／未使用キーも移動しない（順序も保持）
- オプションで:
    --move-unused   未使用キーを末尾「_unused」へ退避
    --prune         未使用キーを削除（※危険、要バックアップ）
    --order code|unicode|keep
                     並べ替え方針（既定: code = コード出現順 / unicode = キーの辞書式昇順 / keep = 変更しない）
    --base ja       欠損補完のベース言語
    --fix           実際に書き換え（なければチェックのみ終了）
- 走査対象:
    webui/ 配下（再帰）、リポジトリ直下の endframe*.py / endframe_*/*.py / oneframe_oichi*.py
- 拡張子: .py .js .ts .tsx .jsx .html（--ext で追加可）
- .git ディレクトリおよび .git というファイル名は常に除外

使用例:
    # チェックのみ（差分があれば exit 1）
    python tools/i18n_check.py

    # 実修正（安全志向: 未使用は動かさない／削除しない）
    python tools/i18n_check.py --fix

    # 本番前に“コード順”へ統一し、未使用は _unused へ退避したい場合
    python tools/i18n_check.py --fix --order code --move-unused

    # 完全クリーン（未使用削除）をやる場合 ※危険・要レビュー
    python tools/i18n_check.py --fix --prune
"""

from __future__ import annotations
import argparse, json, os, re, sys, tempfile
from pathlib import Path
from typing import Dict, List, Set, Tuple, Iterable, Optional
from collections import OrderedDict

# ---------------- 設定 ----------------
DEFAULT_LOCALES_DIR = "webui/locales"
DEFAULT_INCLUDE_EXTS = {".py", ".js", ".ts", ".tsx", ".jsx", ".html"}
DEFAULT_DIR_TARGETS = ["webui"]
DEFAULT_GLOB_TARGETS = [
    "endframe*.py",
    "endframe_*/*.py",
    "oneframe_oichi*.py",
]
DEFAULT_EXCLUDE_DIRS = {".git"}

KEY_PATTERNS = [
    # i18n キーを抽出するための正規表現パターン群
    r"""(?<!\w)(?:t|tr|translate|_|i18n\.t)\(\s*["']([^"']+)["']\s*[),]""",
    r"""i18n\.t\(\s*["']([^"']+)["']\s*\)""",
    r"""(?<!\w)t\(\s*["']([^"']+)["']\s*\)""",
    r"""{{\s*t\(\s*["']([^"']+)["']\s*\)\s*}}""",
]

# JSONC のコメントを取り除くためのパターン
_COMMENT_LINE = re.compile(r"^\s*//.*$")
_COMMENT_BLOCK = re.compile(r"/\*.*?\*/", re.DOTALL)


def strip_jsonc(text: str) -> str:
    """JSONC 形式の文字列からコメントを除去し、純粋な JSON に変換する。"""

    text = _COMMENT_BLOCK.sub("", text)
    return "\n".join(line for line in text.splitlines() if not _COMMENT_LINE.match(line))


def read_json_map(path: Path) -> Dict[str, object]:
    """指定されたパスから JSON ファイルを読み込み、辞書として返す。

    JSONC コメントを除去したうえで読み込む。キーに前後空白が含まれている
    場合はトリムして正規化し、ネストされた辞書も同様に処理する。
    読み込んだ内容がオブジェクトでない場合は :class:`ValueError` を送出。
    """

    data = json.loads(strip_jsonc(path.read_text(encoding="utf-8")) or "{}")
    if not isinstance(data, dict):
        raise ValueError(f"{path} is not a JSON object")

    def _normalize(obj: Dict[str, object]) -> Dict[str, object]:
        """キーの前後空白を削除しつつ再帰的に辞書を正規化する。"""

        normalized: OrderedDict[str, object] = OrderedDict()
        for k, v in obj.items():
            key = k.strip()
            if isinstance(v, dict):
                v = _normalize(v)
            normalized[key] = v
        return normalized

    return _normalize(data)


def json_dumps_canonical(mapping: Dict[str, object]) -> str:
    """キーの並び順を保持したまま比較用の最小 JSON 文字列を生成する。"""

    # 比較用: ensure_ascii=False + インデント無し + キー順はそのまま
    return json.dumps(mapping, ensure_ascii=False, separators=(",", ":"), sort_keys=False)


def write_atomic(path: Path, text: str) -> None:
    """テンポラリに書き出してから `os.replace` で原子的にファイルを置換する。"""

    # 同一ディレクトリにテンポラリ → os.replace で原子的置換（同一FS前提）
    # Windows でも動作（MoveFileEx 互換）。:contentReference[oaicite:4]{index=4}
    tmp_dir = path.parent
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=tmp_dir, delete=False) as tf:
        tf.write(text)
        tmp_name = tf.name
    os.replace(tmp_name, path)  # atomic

# ---------- 走査 ----------
def iter_files_dir(root: Path, include_exts: Set[str], exclude_dirs: Set[str]) -> Iterable[Path]:
    """対象ディレクトリを再帰的に走査し、指定拡張子のファイルを列挙する。"""

    for cur_root, dirs, files in os.walk(root, topdown=True):
        # .git 等はここで枝刈り（下位へ潜らせない）:contentReference[oaicite:5]{index=5}
        dirs[:] = [d for d in dirs if d not in exclude_dirs and d != ".git"]
        for fn in files:
            if fn == ".git":
                continue
            p = Path(cur_root, fn)
            if p.suffix.lower() in include_exts:
                yield p


def iter_files_glob(repo_root: Path, patterns: List[str]) -> Iterable[Path]:
    """グロブパターンに一致するファイルを列挙する。"""

    for pat in patterns:
        for p in repo_root.glob(pat):
            if p.is_file():
                yield p


def collect_code_files(repo_root: Path,
                       dir_targets: List[str],
                       glob_targets: List[str],
                       include_exts: Set[str],
                       exclude_dirs: Set[str]) -> List[Path]:
    """解析対象となるコードファイル一覧を取得する。"""

    files: Set[Path] = set()
    for d in dir_targets:
        base = repo_root / d
        if base.is_dir():
            files.update(iter_files_dir(base, include_exts, exclude_dirs))
    files.update(iter_files_glob(repo_root, glob_targets))
    return sorted(files, key=lambda p: str(p).replace("\\", "/"))


def extract_keys_from_text(text: str) -> List[str]:
    """ソースコード文字列から i18n キーを抽出する。"""

    keys: List[str] = []
    for pat in KEY_PATTERNS:
        for m in re.finditer(pat, text):
            k = (m.group(1) or "").strip()
            if k:
                keys.append(k)
    return keys


def collect_used_keys(files: List[Path]) -> List[str]:
    """コードファイル群を走査して使用されているキーの順序リストを得る。"""

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

# ---------- locales ----------
def find_locale_files(locales_dir: Path) -> Dict[str, Path]:
    """ロケールディレクトリ内の JSON ファイルを探索し、言語コードをキーにした辞書を返す。"""

    result: Dict[str, Path] = {}
    for p in sorted(locales_dir.glob("*.json")):
        lang = p.stem.strip()
        if lang:
            result[lang] = p
    return result


# ---------- 並べ替え ----------
def build_expected_map(*,
    current: Dict[str, object],
    used_order: List[str],
    base_map: Dict[str, object],
    order_mode: str,          # "code" | "unicode" | "keep"
    move_unused: bool,
    prune: bool
) -> OrderedDict:
    """現在のロケールと使用キーから期待されるマップを生成する。

    Parameters
    ----------
    current : Dict[str, object]
        現在のロケール辞書。
    used_order : List[str]
        ソースコードから取得したキーの使用順序。
    base_map : Dict[str, object]
        欠損値を補完するときに参照するベース言語の辞書。
    order_mode : str
        並べ替え方針。"code" / "unicode" / "keep" のいずれか。
    move_unused : bool
        True の場合、未使用キーを `_unused` へ退避する。
    prune : bool
        True の場合、未使用キーを完全に削除する（危険）。

    Returns
    -------
    OrderedDict
        生成したロケールマップ。
    """

    cur_keys = [k for k in current.keys() if k != "_unused"]
    cur_set  = set(cur_keys)
    used_set = set(used_order)

    # 欠損は base から補完（なければキー名をそのまま値に）
    fixed = dict(current)
    for k in used_order:
        if k not in fixed:
            fixed[k] = base_map.get(k, k)

    # 並べ替え
    if order_mode == "keep":
        # 現在の並びを保持（不足分は末尾に追加）
        od = OrderedDict((k, fixed[k]) for k in cur_keys if k in fixed)
        for k in used_order:
            if k not in od:
                od[k] = fixed[k]
    elif order_mode == "unicode":
        all_keys = [k for k in fixed.keys() if k != "_unused"]
        od = OrderedDict((k, fixed[k]) for k in sorted(all_keys))
    else:  # "code" 既定：コード出現順 → 残りは現状順を維持
        od = OrderedDict()
        for k in used_order:
            od[k] = fixed[k]
        # 使われていない既存キーは「いまの相対順」を保って末尾へ
        for k in cur_keys:
            if k not in od:
                od[k] = fixed[k]

    # 未使用キーの扱い
    if prune:
        # 完全削除（危険）。差分レビュー前提でのみ使用推奨
        for k in list(od.keys()):
            if k not in used_set:
                del od[k]
    elif move_unused:
        # _unused へ退避（現状順を維持）
        unused = OrderedDict((k, od.pop(k)) for k in list(od.keys()) if k not in used_set)
        if unused:
            od["_unused"] = unused
        elif "_unused" in current:
            # 既存の _unused は消す（空なら不要）
            pass
    else:
        # 移動しない：もし既存に _unused があって new に無ければ、そのまま残す
        if "_unused" in current:
            od["_unused"] = current["_unused"]

    return od


def diff_summary(before: Dict[str, object], after: OrderedDict) -> Tuple[bool, Dict[str, int]]:
    """before/after を比較し、差分の有無と統計情報を返す。"""

    # 「キー集合・順序・値」の完全一致かをテキスト化して判定
    a = json_dumps_canonical(before)
    b = json_dumps_canonical(after)
    changed = (a != b)

    # 参考：件数内訳
    before_keys = [k for k in before.keys() if k != "_unused"]
    after_keys  = [k for k in after.keys()  if k != "_unused"]
    added   = len([k for k in after_keys if k not in before_keys])
    removed = len([k for k in before_keys if k not in after_keys])
    reordered = int(not changed and before_keys != after_keys)  # テキスト一致なら 0

    return changed, {"added": added, "removed": removed, "reordered_hint": reordered}


# ---------- main ----------
def main() -> None:
    """コマンドライン引数を解析し、i18n チェックまたは修正を実行する。"""

    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="repo root")
    ap.add_argument("--locales", default=DEFAULT_LOCALES_DIR, help="locales dir")
    ap.add_argument("--base", default="ja", help="base language for filling missing (default: ja)")
    ap.add_argument("--order", choices=["code","unicode","keep"], default="code", help="reordering policy")
    ap.add_argument("--move-unused", action="store_true", help="move unused keys to _unused (do not delete)")
    ap.add_argument("--prune", action="store_true", help="delete unused keys (DANGEROUS; review diff!)")
    ap.add_argument("--fix", action="store_true", help="apply changes atomically (otherwise check only)")
    ap.add_argument("--exclude-dir", action="append", default=[], help="additional dirs to exclude (repeatable)")
    ap.add_argument("--ext", action="append", default=[], help="additional file extensions (e.g. .vue)")
    ap.add_argument("--no-dir-target", action="store_true", help="ignore default dir targets")
    ap.add_argument("--no-glob-target", action="store_true", help="ignore default glob targets")
    args = ap.parse_args()

    repo_root = Path(args.root).resolve()
    locales_dir = (repo_root / args.locales)
    if not locales_dir.exists():
        print(f"[ERROR] locales dir not found: {locales_dir}", file=sys.stderr)
        sys.exit(2)

    include_exts = set(DEFAULT_INCLUDE_EXTS)
    for e in args.ext:
        if not e.startswith("."): e = "." + e
        include_exts.add(e.lower())

    exclude_dirs = set(DEFAULT_EXCLUDE_DIRS) | set(args.exclude_dir)

    dir_targets  = [] if args.no_dir_target  else list(DEFAULT_DIR_TARGETS)
    glob_targets = [] if args.no_glob_target else list(DEFAULT_GLOB_TARGETS)

    # ---- Phase 1: 全コード走査（まだ一切書き換えない）----
    code_files = collect_code_files(repo_root, dir_targets, glob_targets, include_exts, exclude_dirs)
    used_order = collect_used_keys(code_files)
    used_set   = set(used_order)

    print("\nScan Summary")
    print("-----------")
    print(f"Code files scanned : {len(code_files)}")
    print(f"Used keys detected : {len(used_order)}")

    # ---- ロケール列挙＆読込（将来言語も自動対応）----
    locale_files = find_locale_files(locales_dir)
    if not locale_files:
        print(f"[ERROR] no locale files (*.json) under {locales_dir}", file=sys.stderr)
        sys.exit(2)
    print(f"Locales detected   : {', '.join(sorted(locale_files.keys()))}")

    locales: Dict[str, Dict[str, object]] = {}
    for lang, p in locale_files.items():
        try:
            mp = read_json_map(p)
            locales[lang] = mp
        except Exception as e:
            print(f"[ERROR] {lang}: {e}", file=sys.stderr)
            sys.exit(2)

    base_lang = args.base if args.base in locales else sorted(locale_files.keys())[0]
    base_map = locales.get(base_lang, {})

    # ---- Phase 2: 期待結果をメモリ上で全言語ぶん作成し、差分比較 ----
    planned_writes: List[Tuple[str, Path, OrderedDict, Dict[str,int]]] = []
    print("\nPer-Locale Report")
    print("-----------------")
    inconsistent = False

    for lang in sorted(locales.keys()):
        cur = locales[lang]
        # 欠損（使用キーに対して無い）を確認
        missing_used = [k for k in used_order if k not in cur]
        extra_used   = sorted(set([k for k in cur.keys() if k != "_unused"]) - used_set)

        # 期待結果を組み立て（未使用キーは既定でそのまま／順序保持）
        expected = build_expected_map(
            current=cur, used_order=used_order, base_map=base_map,
            order_mode=args.order, move_unused=args.move_unused, prune=args.prune
        )

        changed, summary = diff_summary(cur, expected)

        print(f"[{lang}] missing(used)={len(missing_used)} extra(used)={len(extra_used)} "
              f"added={summary['added']} removed={summary['removed']}")

        if changed:
            inconsistent = True
            planned_writes.append((lang, locale_files[lang], expected, summary))

    # ---- Phase 3: 書き換え（--fix のときだけ／差分がある言語だけ／原子的置換）----
    print("\nResult")
    print("------")
    if planned_writes:
        print(f"❗ Pending changes: {len(planned_writes)} locale file(s)")
        if args.fix:
            for lang, path, expected, summary in planned_writes:
                text = json.dumps(expected, ensure_ascii=False, indent=2)
                if not text.endswith("\n"): text += "\n"
                write_atomic(path, text)
                print(f"  [APPLIED] {path}  (+{summary['added']} / -{summary['removed']})")
            print("✅ Applied atomically.")
            sys.exit(0)
        else:
            print("Run with --fix to apply them (no files were modified).")
            sys.exit(1)
    else:
        print("✅ All locales consistent (no write).")
        sys.exit(0)


if __name__ == "__main__":
    main()

