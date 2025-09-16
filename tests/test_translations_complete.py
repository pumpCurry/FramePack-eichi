import json
import logging
from pathlib import Path

import pytest

from webui.locales.i18n import init, test_translate as translate_checker

def test_all_locale_keys_have_translations(caplog: pytest.LogCaptureFixture) -> None:
    """
    すべてのロケール JSON ファイルを対象に、各キーが全言語で翻訳されているか検証するテスト。

    - webui/locales ディレクトリ内の *.json ファイルを列挙し、そのファイル名からロケール名を取得します。
    - 各 JSON ファイルのすべてのキーに対し test_translate(key, locales) を実行し、
      翻訳が存在するかを判定します。
    - いずれかの言語で翻訳が欠落している場合は failure_counts に記録し、
      最終的に failure_total が 0 であることをアサーションします。
    - caplog 引数を用いることで、pytest 実行時にログ出力を確認することも可能です。
    """

    # 必要であれば初期化（デフォルト言語を設定）
    init("ja")

    locales_dir = Path("webui/locales")
    # locales_dir から *.json をすべて取得してソート
    json_files = sorted(locales_dir.glob("*.json"))
    # ファイル名 (en.json -> 'en') からロケール名を抽出
    locales = [p.stem for p in json_files]

    # 各ロケールごとの翻訳欠落数を記録する辞書
    failure_counts: dict[str, int] = {loc: 0 for loc in locales}
    success_total: int = 0
    failure_total: int = 0

    # 各 JSON ファイルごとに処理
    for json_path in json_files:
        # JSON ファイルを読み込む
        data = json.loads(json_path.read_text(encoding="utf-8"))
        # 各キーを取得し、全ロケールで翻訳されているか判定
        for key in data.keys():
            all_present, _, results = translate_checker(key, locales)
            if all_present:
                # すべての言語で翻訳が存在した場合
                success_total += 1
            else:
                # どこかの言語で翻訳が欠落している場合
                failure_total += 1
                for locale, _, value in results:
                    if value is None:
                        # 該当ロケールの欠落数をカウント
                        failure_counts[locale] += 1

    # ログ出力（pytest の caplog フィクスチャを通して確認可能）
    logger = logging.getLogger(__name__)
    logger.info(
        "Translation test completed: success=%s, failure=%s, failures_by_locale=%s",
        success_total,
        failure_total,
        failure_counts,
    )

    # 1つでも欠落している場合はテストを失敗させる
    assert failure_total == 0, f"Missing translations: {failure_counts}"
