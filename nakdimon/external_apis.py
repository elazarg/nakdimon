from __future__ import annotations

import logging
from collections import defaultdict, Counter
import re
import json
from functools import wraps

import requests
from cachier import cachier

from hebrew import Niqqud
import hebrew


class DottingError(RuntimeError):
    pass


def split_string_by_length(text: str, maxlen) -> list[str]:
    return [''.join(s).strip() for s in hebrew.split_by_length(text, maxlen)]


def piecewise(maxlen):
    def inner(fetch):
        @wraps(fetch)
        def fetcher(text):
            return ' '.join(fetch(chunk) for chunk in split_string_by_length(text, maxlen))
        return fetcher
    return inner


def fix_snopi(dotted_text: str, undotted_text: str) -> str:
    items = list(hebrew.iterate_dotted_text(dotted_text))
    for i in range(len(undotted_text)):
        if len(items) == i:
            items.insert(i, hebrew.HebrewItem(undotted_text[i], '', '', '', ''))
        elif undotted_text[i] != ' ' and items[i].letter == ' ':
            del items[i]
        elif undotted_text[i] != items[i].letter:
            items.insert(i, hebrew.HebrewItem(undotted_text[i], '', '', '', ''))
    return hebrew.items_to_text(items)


@cachier()
@piecewise(75)  # estimated maximum for reasonable time
def fetch_snopi(undotted_text: str) -> str:
    # Add bogus continuation in case there's only a single word
    # so Snopi will not decide to answer with single-word-analysis
    logging.debug(repr(undotted_text))
    dummy = False
    if ' ' not in undotted_text:
        dummy = True
        undotted_text = undotted_text + ' 1'

    url = 'http://www.nakdan.com/GetResult.aspx'

    payload = {
        "txt": undotted_text,
        "ktivmale": 'true',
    }
    headers = {
        'Referer': 'http://www.nakdan.com/nakdan.aspx',
    }

    r = requests.post(url, data=payload, headers=headers)
    r.raise_for_status()

    dotted_text = r.text.strip().split('Result')[1][1:-2]
    logging.debug(repr(dotted_text))
    if hebrew.remove_niqqud(dotted_text) != undotted_text:
        logging.debug('Fixing...')
        dotted_text = fix_snopi(dotted_text, undotted_text)
        logging.debug(repr(dotted_text))
    assert hebrew.remove_niqqud(dotted_text) == undotted_text, f'{repr(dotted_text)}\n!=\n{repr(undotted_text)}'
    if dummy:
        dotted_text = dotted_text[:-2]
    return dotted_text


@cachier()
@piecewise(100)
def fetch_morfix(text: str) -> str:
    import sys
    print('Error: Morfix does not allow automatic diacritization! exiting.', file=sys.stderr)
    exit(1)

    url = 'https://nakdan.morfix.co.il/nikud/NikudText'

    payload = {
        "text": text,
        "isLogged": 'false',
    }
    headers = {
    }

    r = requests.post(url, data=payload, headers=headers)
    r.raise_for_status()
    return json.loads(r.json()['nikud'])['OutputText']


@cachier()
@piecewise(10000)
def fetch_dicta(text: str) -> str:
    text = '\n'.join(line for line in text.split('\n') if not line.startswith('https') and not line.startswith('#')).strip()
    def extract_word(k):
        if k['options']:
            res = k['options'][0][0]
            res = res.replace('|', '')
            res = res.replace(Niqqud.KUBUTZ + 'ו' + Niqqud.METEG, 'ו' + Niqqud.SHURUK)
            res = res.replace(Niqqud.HOLAM + 'ו' + Niqqud.METEG, 'ו' + Niqqud.HOLAM)
            res = res.replace(Niqqud.METEG, '')

            res = re.sub(Niqqud.KAMATZ + 'ו' + '(?=[א-ת])', 'ו' + Niqqud.HOLAM, res)
            res = res.replace(Niqqud.REDUCED_KAMATZ + 'ו', 'ו' + Niqqud.HOLAM)

            res = res.replace(hebrew.DAGESH_LETTER * 2, hebrew.DAGESH_LETTER)
            res = res.replace('\u05be', '-')
            res = res.replace('יְהוָֹה', 'יהוה')
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
    r.raise_for_status()
    result = ''.join(extract_word(k) for k in r.json())
    if len(hebrew.find_longest_undotted(result)) > 40:
        raise DottingError('Failed to dot')
    return result


def fetch_dicta_version() -> str:
    url = 'https://nakdan-4-0.loadbalancer.dicta.org.il/api/'
    r = requests.get(url + 'debug')
    r.raise_for_status()
    model_version = r.json()['version'].split("\n")[4].strip()

    r = requests.get(url + 'wlver')
    r.raise_for_status()
    wordlist_version = r.json()['version']

    return f'{model_version}, {wordlist_version}'


def make_fetch_nakdimon(path):
    @piecewise(10000)
    def fetch_nakdimon(text: str) -> str:
        url = 'http://127.0.0.1:5000'

        payload = {
            "text": text,
            "model_name": path,
        }
        headers = {
        }

        r = requests.post(url, data=payload, headers=headers)
        r.raise_for_status()
        return r.text
    return fetch_nakdimon


SYSTEMS = {
    'Snopi': fetch_snopi,  # Too slow
    'Morfix': fetch_morfix,  # terms-of-use issue
    'Dicta': fetch_dicta,
    'Nakdimon': make_fetch_nakdimon('final_model/final.h5'),
}
all_oov = set()


class MajorityDiacritizer:
    dictionary: dict[str, str]

    @staticmethod
    def update_possibilities(possibilities: defaultdict[str, Counter], train_paths: tuple[str, ...]) -> None:
        import hebrew
        for token in hebrew.collect_tokens(train_paths):
            word = token.to_text()
            if word:
                left, word, right = hebrew.split_nonhebrew(word)
                if word:
                    possibilities[hebrew.remove_niqqud(word)][word] += 1
                    for t in token.items:
                        if hebrew.is_hebrew_letter(t.letter):
                            possibilities[t.letter][t.to_text()] += 1

    def __init__(self, possibilities: defaultdict[str, Counter]) -> None:
        self.dictionary = {word: options.most_common(1)[0][0]
                           for word, options in possibilities.items()}

    def is_oov(self, word: str) -> bool:
        left, word, right = hebrew.split_nonhebrew(hebrew.remove_niqqud(word))
        if word not in self.dictionary:
            # all_oov.add(word)
            return True
        return False

    def diacritize_token(self, word: str) -> str:
        left, word, right = hebrew.split_nonhebrew(word)
        word = hebrew.remove_niqqud(word)
        if word in self.dictionary:
            result = self.dictionary[word]
        else:
            result = word  # ''.join([self.dictionary.get(letter, '') for letter in word])
        return left + result + right

    def __call__(self, text: str) -> str:
        return ' '.join([self.diacritize_token(token) for token in hebrew.remove_niqqud(text).split()])


MAJ_MOD = 'MajMod'
MAJ_ALL_NO_DICTA = 'MajAllNoDicta'
MAJ_ALL_WITH_DICTA = 'MajAllWithDicta'


@cachier()
def prepare_majority():
    logging.info('Preparing MajMod...')
    possibilities = defaultdict(Counter)
    res = {}

    MajorityDiacritizer.update_possibilities(possibilities, tuple([
        'hebrew_diacritized/modern/'
    ]))
    res[MAJ_MOD] = MajorityDiacritizer(possibilities)

    logging.info('Preparing MajAllNoDicta...')
    MajorityDiacritizer.update_possibilities(possibilities, tuple([
        'hebrew_diacritized/poetry',
        'hebrew_diacritized/rabanit',
        'hebrew_diacritized/pre_modern',
        'hebrew_diacritized/shortstoryproject_predotted',
        'hebrew_diacritized/shortstoryproject_Dicta',
        'hebrew_diacritized/validation'
    ]))
    res[MAJ_ALL_NO_DICTA] = MajorityDiacritizer(possibilities)

    logging.info('Preparing MajAllWithDicta...')
    MajorityDiacritizer.update_possibilities(possibilities, ('hebrew_diacritized/dictaTestCorpus',))
    res[MAJ_ALL_WITH_DICTA] = MajorityDiacritizer(possibilities)

    logging.info('Done preparing.')
    return res


def fetch_dicta_count_ambiguity(text: str):
    url = 'https://nakdan-2-0.loadbalancer.dicta.org.il/api'

    payload = {
        "task": "nakdan",
        "genre": "modern",
        "data": text,
        # "addmorph": True,
        "keepqq": False,
        "nodageshdefmem": False,
        "patachma": False,
        "keepmetagim": True,
    }
    headers = {
        'content-type': 'text/plain;charset=UTF-8'
    }

    r = requests.post(url, json=payload, headers=headers)
    r.raise_for_status()
    return [len(set(token['options'])) for token in r.json() if not token['sep']]

# fetch_snopi.clear_cache()
# fetch_nakdimon_fullnew.clear_cache()
# fetch_dicta.clear_cache()
# prepare_majority.clear_cache()


if __name__ == '__main__':
    SYSTEMS.update(prepare_majority())
