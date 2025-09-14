import json
from pathlib import Path

LOCALES = Path(__file__).resolve().parents[1] / 'webui' / 'locales'
LANGS = ['ja', 'en', 'zh-tw', 'ru']

def load_keys(lang):
    with open(LOCALES / f'{lang}.json', encoding='utf-8') as f:
        data = json.load(f, object_pairs_hook=list)
    return [k for k,_ in data]

def test_locale_keys_match():
    base = load_keys('ja')
    for lang in LANGS[1:]:
        assert load_keys(lang) == base, f'{lang} locale keys do not match ja locale'
