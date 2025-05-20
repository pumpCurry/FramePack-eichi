from locales.i18n_extended import translate
from enum import Enum


# 結合箇所
class COMBINE_MODE(Enum):
    FIRST = 0
    LAST = 1


# 検証用なので後方のみ
COMBINE_MODE_OPTIONS = {
    # 先頭側への結合評価はあまり必要ないと思われるのでコメント
    # translate("テンソルデータの前方（先頭）"): COMBINE_MODE.FIRST,
    translate("テンソルデータの後方（末尾）"): COMBINE_MODE.LAST,
}
COMBINE_MODE_OPTIONS_KEYS = list(COMBINE_MODE_OPTIONS.keys())
COMBINE_MODE_DEFAULT = COMBINE_MODE_OPTIONS_KEYS[0]


def get_combine_mode(combine_mode):
    """COMBINE_MODEのEnumからCOMBINE_MODE_OPTIONSのキーの値を取得する

    Args:
        combine_mode (COMBINE_MODE): 結合モードのEnum値

    Returns:
        str: COMBINE_MODE_OPTIONSのキー。見つからない場合はデフォルト値
    """
    for key, value in COMBINE_MODE_OPTIONS.items():
        if value == combine_mode:
            return key
    return COMBINE_MODE_DEFAULT  # 見つからない場合はデフォルト値を返す
