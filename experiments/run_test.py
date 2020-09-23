from pathlib import Path

import utils

from experiments import external_apis
import hebrew
from experiments.metrics import *



SYSTEMS = {
    'Snopi': external_apis.fetch_snopi,  # Too slow
    'Morfix': external_apis.fetch_morfix,  # terms-of-use issue
    'Nakdan': external_apis.fetch_dicta,
    'Nakdimon': external_apis.fetch_nakdimon,
}


def diacritize(sysname, filename):
    assert 'test' in Path(filename).parts

    with open(filename, 'r', encoding='utf8') as f:
        expected = f.read()
    cleaned = hebrew.remove_niqqud(expected)
    actual = SYSTEMS[sysname](cleaned)

    outfile = f'actual/{sysname}' / Path(filename).relative_to('test/')
    Path(outfile).parent.mkdir(parents=True, exist_ok=True)

    with open(outfile, 'w', encoding='utf8') as f:
        f.write(actual)


def diacritize_all(sysname):
    for filename in utils.iterate_files(['test']):
        diacritize(sysname, filename)


def format_latex(all_results):
    for system, results in all_results.items():
        print('{system} & {cha:.2%} & {dec:.2%} & {wor:.2%} & {voc:.2%} \\\\'.format(system=system, **results)
              .replace('%', ''))


def collect_metrics(actual, expected):
    return {
        'cha': metric_cha(actual, expected),
        'dec': metric_dec(actual, expected),
        'wor': metric_wor(actual, expected),
        'voc': metric_wor(actual, expected, vocalize=True)
    }


def run(filename):
    with open(filename, 'r', encoding='utf8') as f:
        expected = f.read()
    cleaned = hebrew.remove_niqqud(expected)
    all_results = {system: collect_metrics(api(cleaned), expected)
                   for system, api in SYSTEMS.items()}
    format_latex(all_results)


if __name__ == '__main__':
    diacritize_all('Nakdan')
