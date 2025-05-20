import json
import os.path

# デフォルト言語設定
# 注意: init関数を呼び出すまでは翻訳機能は使用できません
lang = "ja"  # 明示的にデフォルト言語を日本語(ja)に設定
translateContext = None

class I18nString(str):
    def __new__(cls, value):
        result = translateContext.get(lang, {}).get(value, value)
        return result

    def __init__(self, value):
        if isinstance(value, I18nString):
            self.add_values = value.add_values
            self.radd_values = value.radd_values
        else:
            self.add_values = []
            self.radd_values = []

    def __str__(self):
        result = translateContext.get(lang, {}).get(self, super().__str__())

        for v in self.radd_values:
            result = str(v) + result

        for v in self.add_values:
            result = result + str(v)

        # hotfix, remove unexpected single quotes
        while len(result) >= 2 and result.startswith("'") and result.endswith("'"):
            result = result[1:-1]

        return result

    def __add__(self, other):
        v = str(self)
        if isinstance(v, I18nString):
            self.add_values.append(other)
            return self
        return v.__add__(other)

    def __radd__(self, other):
        v = str(self)
        if isinstance(v, I18nString):
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

def translate(key: str):
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
    
    return I18nString(key)

def load_translations():
    translations = {}
    locales_dir = os.path.join(os.path.dirname(__file__), './')

    for locale in ["en", "ja", "zh-tw", "ru"]:
        json_file = os.path.join(locales_dir, f"{locale}.json")
        if os.path.exists(json_file):
            with open(json_file, 'r', encoding='utf-8') as f:
                translations[locale] = json.load(f)
        else:
            print(f"Warning: Translation file {json_file} not found")
            translations[locale] = {}

    return translations

def init(locale="ja"):
    """言語を初期化します。
    
    Args:
        locale: 使用する言語コード（例: 'ja', 'en', 'zh-tw'）。
               未対応の言語の場合は自動的に'ja'が使用されます。
    """
    global lang
    global translateContext
    
    # 対応言語のリスト
    supported_locales = ["ja", "en", "zh-tw", "ru"]
    
    # 対応していない言語の場合はデフォルト言語(ja)を使用
    if locale not in supported_locales:
        print(f"[WARNING] Unsupported language: {locale}. Falling back to 'ja'")
        locale = "ja"
    
    lang = locale
    translateContext = load_translations()
