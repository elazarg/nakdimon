from typing import List
import requests
import json
from functools import wraps
from cachier import cachier

from hebrew import Niqqud
import hebrew


def split_string_by_length(text: str, maxlen) -> List[str]:
    return [''.join(s).strip() for s in hebrew.split_by_length(text, maxlen)]


def piecewise(maxlen):
    def inner(fetch):
        @wraps(fetch)
        def fetcher(text):
            return ' '.join(fetch(chunk) for chunk in split_string_by_length(text, maxlen))
        return fetcher
    return inner


@cachier()
@piecewise(75)  # estimated maximum for reasonable time
def fetch_snopi(text: str) -> str:
    # Add bogus continuation in case there's only a single word
    # so Snopi will not decide to answer with single-word-analysis
    text = text + ' 1'

    url = 'http://www.nakdan.com/GetResult.aspx'

    payload = {
        "txt": text,
        "ktivmale": 'true',
    }
    headers = {
        'Referer': 'http://www.nakdan.com/nakdan.aspx',
    }

    r = requests.post(url, data=payload, headers=headers)
    res = list(r.text.split('Result')[1][1:-2])
    items = list(hebrew.iterate_dotted_text(res))

    for i in range(len(text)):
        if text[i] != ' ' and items[i].letter == ' ':
            del items[i]
        elif text[i] != items[i].letter:
            items.insert(i, hebrew.HebrewItem(text[i], '', '', '', ''))
    res = hebrew.items_to_text(items)
    assert hebrew.remove_niqqud(res) == text

    return res[:-2]


@cachier()
@piecewise(100)
def fetch_morfix(text: str) -> str:
    url = 'https://nakdan.morfix.co.il/nikud/NikudText'

    payload = {
        "text": text,
        "isLogged": 'false',
    }
    headers = {
    }

    r = requests.post(url, data=payload, headers=headers)
    return json.loads(r.json()['nikud'])['OutputText']


@cachier()
@piecewise(10000)
def fetch_dicta(text: str) -> str:
    def extract_word(k):
        if k['options']:
            res = k['options'][0][0]
            res = res.replace('|', '')
            res = res.replace(Niqqud.KUBUTZ + 'ו' + Niqqud.METEG, 'ו' + Niqqud.SHURUK)
            res = res.replace(Niqqud.HOLAM + 'ו' + Niqqud.METEG, 'ו' + Niqqud.HOLAM)
            res = res.replace(Niqqud.METEG, '')
            return res
        return k['word']

    url = 'https://nakdan-2-0.loadbalancer.dicta.org.il/api'

    payload = {
        "task": "nakdan",
        "genre": "modern",
        "data": text,
        "addmorph": True,
        "keepqq": False,
        "nodageshdefmem": False,
        "patachma": False,
        "keepmetagim": True,
    }
    headers = {
        'content-type': 'text/plain;charset=UTF-8'
    }

    r = requests.post(url, json=payload, headers=headers)
    return ''.join(extract_word(k) for k in r.json())


@cachier()
@piecewise(10000)
def fetch_nakdimon(text: str) -> str:
    url = 'http://127.0.0.1:5000'

    payload = {
        "text": text,
    }
    headers = {
    }

    r = requests.post(url, data=payload, headers=headers)
    return r.text


if __name__ == '__main__':
    text = 'ה"קפיטליסטית" של סוף המאה ה-19, ומהוות מופת לפעולה וולונטרית שאינה'
    print(fetch_nakdimon(text))
