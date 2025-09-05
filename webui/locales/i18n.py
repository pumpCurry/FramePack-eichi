import json
import os.path
from typing import List, Optional, Tuple

# デフォルト言語設定
# 注意: init関数を呼び出すまでは翻訳機能は使用できません
lang = "ja"  # 明示的にデフォルト言語を日本語(ja)に設定
translateContext = None

class I18nString(str):
    def __new__(cls, value: str):
        # 翻訳コンテキストと現在の言語を参照する
        global translateContext, lang
        # translateContext が None の場合は init が呼ばれていない可能性がある
        # __new__ 内では translateContext が None でもエラーにはならず、辞書の get でキーが無い場合は value を返す
        result = translateContext.get(lang, {}).get(value, value)
        # 親クラスの __new__ で文字列を生成
        return str.__new__(cls, result)

    def __init__(self, value):
        # I18nString同士を連結するための付加値リストを保持
        if isinstance(value, I18nString):
            self.add_values = value.add_values
            self.radd_values = value.radd_values
        else:
            self.add_values = []
            self.radd_values = []

    def __str__(self):
        # 現在の言語での翻訳文字列を取得
        result = translateContext.get(lang, {}).get(self, super().__str__())

        # 右側に追加する値を先に処理
        for v in self.radd_values:
            result = str(v) + result

        # 左側に追加する値を後に処理
        for v in self.add_values:
            result = result + str(v)

        # hotfix, remove unexpected single quotes
        while len(result) >= 2 and result.startswith("'") and result.endswith("'"):
            result = result[1:-1]

        return result

    def __add__(self, other):
        v = str(self)
        if isinstance(v, I18nString):
            # 既に翻訳済みオブジェクトであれば add_values に追加して遅延評価
            self.add_values.append(other)
            return self
        # そうでなければ通常の文字列加算にフォールバック
        return v.__add__(other)

    def __radd__(self, other):
        v = str(self)
        if isinstance(v, I18nString):
            # 右辺からの加算の場合 radd_values に追加
            self.radd_values.append(other)
            return self
        return other.__add__(v)

    def __hash__(self) -> int:
        return super().__hash__()

    def format(self, *args, **kwargs) -> str:
        v = str(self)
        if isinstance(v, I18nString):
            return super().format(*args, **kwargs)
        return v.format(*args, **kwargs)

    def unwrap(self):
        # I18nString を純粋な文字列に変換
        return super().__str__()

    @staticmethod
    def unwrap_strings(obj):
        """Unwrap all keys in I18nStrings in the object"""
        if isinstance(obj, I18nString):
            yield obj.unwrap()
            for v in obj.add_values:
                yield from I18nString.unwrap_strings(v)
            for v in obj.radd_values:
                yield from I18nString.unwrap_strings(v)
            return
        yield obj

def translate(key: str) -> I18nString:
    """指定されたキーに対応する翻訳文字列を返します。
    
    Args:
        key: 翻訳したい文字列のキー
        
    Returns:
        I18nString: 現在の言語設定に基づいた翻訳文字列
    """
    # デバッグ用：translateContextがロードされていない場合に自動的にロード
    global translateContext
    if translateContext is None:
        # 自動的にinitializeを呼び出す
        init(lang)
    # I18nString オブジェクトを返すことで遅延翻訳を実現
    return I18nString(key)

def load_translations() -> dict[str, dict[str, str]]:
    """ロケールディレクトリから利用可能な JSON を読み込み、翻訳テーブルを構築する。"""
    translations: dict[str, dict[str, str]] = {}
    locales_dir = os.path.join(os.path.dirname(__file__), './')

    # ディレクトリ内の *.json ファイルをすべて対象とする
    for file_name in os.listdir(locales_dir):
        if not file_name.endswith('.json'):
            continue
        locale = os.path.splitext(file_name)[0]
        json_file = os.path.join(locales_dir, file_name)
        if os.path.exists(json_file):
            with open(json_file, 'r', encoding='utf-8') as f:
                translations[locale] = json.load(f)
        else:
            print("Warning: Translation file {0} not found".format(json_file))
            translations[locale] = {}

    return translations

def init(locale: str = "ja") -> None:
    """言語を初期化します。
    
    Args:
        locale: 使用する言語コード（例: 'ja', 'en', 'zh-tw', 'ru'）。
               未対応の言語の場合は自動的に'ja'が使用されます。
    """
    global lang
    global translateContext

    # 利用可能なロケールを翻訳データから動的に取得
    available_translations = load_translations()
    supported_locales = set(available_translations.keys())

    # 対応していない言語の場合はデフォルト言語(ja)を使用
    if locale not in supported_locales:
        print("Unsupported language: {0}. Falling back to 'ja'".format(locale))
        locale = "ja"
    
    lang = locale
    translateContext = available_translations

def test_translate(
    key: str,
    languages: Optional[List[str]] = None
) -> Tuple[bool, int, List[Tuple[str, str, Optional[str]]]]:
    """
    指定されたキーが複数の言語で翻訳されているかをテストする。

    Args:
        key: 検証したい翻訳キー。
        languages: テスト対象の言語コードのリスト。未指定の場合はすべての利用可能言語。

    Returns:
        all_present: すべての言語で翻訳が存在したか。
        success_count: 翻訳が存在した言語数。
        results: (言語コード, キー, 翻訳文字列または None) のリスト。
    """
    global lang, translateContext
    # translateContext が未初期化なら初期化
    if translateContext is None:
        init(lang)

    # 利用可能な言語リストを取得
    available = list(translateContext.keys())
    # テスト対象の言語リストを決定
    test_langs = languages if languages is not None else available

    original_lang = lang
    success_count = 0
    results: List[Tuple[str, str, Optional[str]]] = []

    for l in test_langs:
        if l not in available:
            # 未対応の言語なら翻訳なしとしてカウント
            results.append((l, key, None))
            continue
        # 一時的に言語を切り替えて翻訳を取得
        lang = l
        translated = str(translate(key))
        if key in translateContext.get(l, {}):
            results.append((l, key, translated))
            success_count += 1
        else:
            results.append((l, key, None))
    # 元の言語を戻す
    lang = original_lang
    all_present = success_count == len(test_langs)
    return all_present, success_count, results
