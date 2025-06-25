"""
お気に入り設定管理モジュール
プロンプト管理機能を拡張し、各種設定をまとめて保存・読み込みする
"""

import os
import json
from datetime import datetime

from locales.i18n_extended import translate


FAVORITE_FILE = 'favorite_settings.json'


def _get_presets_dir():
    webui_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    presets_dir = os.path.join(webui_path, 'presets')
    os.makedirs(presets_dir, exist_ok=True)
    return presets_dir


def _get_favorite_file_path():
    return os.path.join(_get_presets_dir(), FAVORITE_FILE)


def initialize_favorites():
    path = _get_favorite_file_path()
    if os.path.exists(path):
        return
    data = {"favorites": []}
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_favorites():
    initialize_favorites()
    path = _get_favorite_file_path()
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(translate("❌ 設定読み込みエラー: {0}").format(e))
        return {"favorites": []}


def save_favorite(name: str, settings: dict):
    if not name:
        return translate("名前を入力してください")

    data = load_favorites()
    favorites = data.get("favorites", [])

    exists = False
    for fav in favorites:
        if fav.get("name") == name:
            fav.update(settings)
            fav["timestamp"] = datetime.now().isoformat()
            exists = True
            break
    if not exists:
        entry = settings.copy()
        entry["name"] = name
        entry["timestamp"] = datetime.now().isoformat()
        favorites.append(entry)

    data["favorites"] = favorites

    path = _get_favorite_file_path()
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return translate("設定 '{0}' を保存しました").format(name)
    except Exception as e:
        print(translate("❌ 設定保存エラー: {0}").format(e))
        return translate("保存エラー: {0}").format(e)


def delete_favorite(name: str):
    if not name:
        return translate("設定を選択してください")

    data = load_favorites()
    favorites = data.get("favorites", [])
    new_favs = [f for f in favorites if f.get("name") != name]
    if len(new_favs) == len(favorites):
        return translate("設定 '{0}' が見つかりません").format(name)

    data["favorites"] = new_favs
    path = _get_favorite_file_path()
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return translate("設定 '{0}' を削除しました").format(name)
    except Exception as e:
        return translate("削除エラー: {0}").format(e)

