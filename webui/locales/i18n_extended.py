def set_lang(language):
    """言語設定を行う"""
    from locales import i18n
    i18n.lang = language
    i18n.init(language)  # 言語設定を反映（明示的に言語を渡す）

# i18nモジュールからtranslate関数をインポート
from locales.i18n import translate

"""
FramePack-eichi 拡張i18nモジュール
各言語間の変換と内部キーの管理を行います
"""

import json
import os.path
from locales import i18n

# 逆マッピング用辞書
_reverse_mapping = {
    # 英語→内部キー
    "0.5 seconds (17 frames)": "_KEY_FRAME_SIZE_05SEC",
    "1 second (33 frames)": "_KEY_FRAME_SIZE_1SEC",
    "Normal": "_KEY_MODE_NORMAL", 
    "Normal mode": "_KEY_MODE_NORMAL_FULL",
    "Loop": "_KEY_MODE_LOOP",
    "Loop mode": "_KEY_MODE_LOOP_FULL",
    "1 second": "_KEY_VIDEO_LENGTH_1SEC",
    "2s": "_KEY_VIDEO_LENGTH_2SEC",
    "3s": "_KEY_VIDEO_LENGTH_3SEC",
    "4s": "_KEY_VIDEO_LENGTH_4SEC",
    "6s": "_KEY_VIDEO_LENGTH_6SEC",
    "8s": "_KEY_VIDEO_LENGTH_8SEC",
    "10s": "_KEY_VIDEO_LENGTH_10SEC",
    "12s": "_KEY_VIDEO_LENGTH_12SEC",
    "16s": "_KEY_VIDEO_LENGTH_16SEC",
    "20s": "_KEY_VIDEO_LENGTH_20SEC",
    
    # 中国語→内部キー
    "0.5秒 (17幀)": "_KEY_FRAME_SIZE_05SEC",
    "1秒 (33幀)": "_KEY_FRAME_SIZE_1SEC",
    "常規": "_KEY_MODE_NORMAL",
    "常規模式": "_KEY_MODE_NORMAL_FULL",
    "循環": "_KEY_MODE_LOOP",
    "循環模式": "_KEY_MODE_LOOP_FULL",
    "1秒": "_KEY_VIDEO_LENGTH_1SEC",
    "2秒": "_KEY_VIDEO_LENGTH_2SEC",
    "3秒": "_KEY_VIDEO_LENGTH_3SEC",
    "4秒": "_KEY_VIDEO_LENGTH_4SEC",
    "6秒": "_KEY_VIDEO_LENGTH_6SEC",
    "8秒": "_KEY_VIDEO_LENGTH_8SEC",
    "10秒": "_KEY_VIDEO_LENGTH_10SEC",
    "12秒": "_KEY_VIDEO_LENGTH_12SEC",
    "16秒": "_KEY_VIDEO_LENGTH_16SEC",
    "20秒": "_KEY_VIDEO_LENGTH_20SEC",
    
    # 日本語→内部キー
    "0.5秒 (17フレーム)": "_KEY_FRAME_SIZE_05SEC",
    "1秒 (33フレーム)": "_KEY_FRAME_SIZE_1SEC",
    "通常": "_KEY_MODE_NORMAL",
    "通常モード": "_KEY_MODE_NORMAL_FULL",
    "ループ": "_KEY_MODE_LOOP",
    "ループモード": "_KEY_MODE_LOOP_FULL",
    "1秒": "_KEY_VIDEO_LENGTH_1SEC",
    "2秒": "_KEY_VIDEO_LENGTH_2SEC",
    "3秒": "_KEY_VIDEO_LENGTH_3SEC",
    "4秒": "_KEY_VIDEO_LENGTH_4SEC",
    "6秒": "_KEY_VIDEO_LENGTH_6SEC",
    "8秒": "_KEY_VIDEO_LENGTH_8SEC",
    "10秒": "_KEY_VIDEO_LENGTH_10SEC", 
    "12秒": "_KEY_VIDEO_LENGTH_12SEC",
    "16秒": "_KEY_VIDEO_LENGTH_16SEC",
    "20秒": "_KEY_VIDEO_LENGTH_20SEC",
    
    # ロシア語→内部キー
    "0.5 секунды (17 кадров)": "_KEY_FRAME_SIZE_05SEC",
    "1 секунда (33 кадра)": "_KEY_FRAME_SIZE_1SEC",
    "Нормальный": "_KEY_MODE_NORMAL",
    "Нормальный режим": "_KEY_MODE_NORMAL_FULL",
    "Цикл": "_KEY_MODE_LOOP",
    "Циклический режим": "_KEY_MODE_LOOP_FULL",
    "1 секунда": "_KEY_VIDEO_LENGTH_1SEC",
    "2 сек": "_KEY_VIDEO_LENGTH_2SEC",
    "3 сек": "_KEY_VIDEO_LENGTH_3SEC",
    "4 сек": "_KEY_VIDEO_LENGTH_4SEC",
    "6 сек": "_KEY_VIDEO_LENGTH_6SEC",
    "8 сек": "_KEY_VIDEO_LENGTH_8SEC",
    "10 сек": "_KEY_VIDEO_LENGTH_10SEC",
    "12 сек": "_KEY_VIDEO_LENGTH_12SEC",
    "16 сек": "_KEY_VIDEO_LENGTH_16SEC",
    "20 сек": "_KEY_VIDEO_LENGTH_20SEC",
}

# 内部キーから各言語への変換マップ
_internal_to_lang = {
    "ja": {},
    "en": {},
    "zh-tw": {},
    "ru": {}
}

def init():
    """逆マッピングを初期化"""
    global _reverse_mapping
    global _internal_to_lang
    
    # 各言語ファイルを読み込み
    locales_dir = os.path.join(os.path.dirname(__file__), './')
    for locale in ["en", "ja", "zh-tw", "ru"]:
        json_file = os.path.join(locales_dir, f"{locale}.json")
        if os.path.exists(json_file):
            with open(json_file, 'r', encoding='utf-8') as f:
                translations = json.load(f)
                
            # 内部キー（_KEY_で始まるもの）の逆マッピングを構築
            for key, value in translations.items():
                if key.startswith("_KEY_"):
                    # 逆マッピング: 翻訳文字列→内部キー
                    _reverse_mapping[value] = key
                    # 正マッピング: 内部キー→翻訳文字列
                    _internal_to_lang[locale][key] = value
    
def get_internal_key(translated_text):
    """翻訳された文字列から内部キーを取得"""
    return _reverse_mapping.get(translated_text, translated_text)
    
def get_original_japanese(translated_text):
    """翻訳された文字列から元の日本語を取得"""
    internal_key = get_internal_key(translated_text)
    # 内部キーが見つからない場合は元の文字列を返す
    if internal_key == translated_text:
        return translated_text
        
    # 内部キーから日本語訳を取得
    return _internal_to_lang.get("ja", {}).get(internal_key, translated_text)

def convert_between_languages(text, from_lang, to_lang):
    """ある言語の文字列を別の言語に変換"""
    # 内部キーを取得
    internal_key = get_internal_key(text)
    
    # 内部キーが見つからない場合は元の文字列を返す
    if internal_key == text:
        return text
        
    # 目的言語の翻訳を取得
    return _internal_to_lang.get(to_lang, {}).get(internal_key, text)

# 初期化を実行
init()
