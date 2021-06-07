import collections
from pathlib import Path

import utils

from external_apis import SYSTEMS, fetch_dicta_count_ambiguity
import hebrew

basepath = 'tests/test/expected'


def diacritize(sysname, filename):
    with open(filename, 'r', encoding='utf8') as f:
        expected = f.read()
    cleaned = hebrew.remove_niqqud(expected)
    return SYSTEMS[sysname](cleaned)


def diacritize_all(sysname):
    for filename in utils.iterate_files([basepath]):
        # if filename.endswith(r'\nrg\6.txt') or filename.endswith(r'president\6.txt'):
        #    continue
        print(filename, end=' ' * 30 + '\r', flush=True)

        actual = diacritize(sysname, filename)

        outfile = filename.replace('expected', sysname)
        Path(outfile).parent.mkdir(parents=True, exist_ok=True)

        with open(outfile, 'w', encoding='utf8') as f:
            f.write(actual)


def count_all_ambiguity():
    c = collections.Counter()
    for filename in utils.iterate_files([basepath]):
        print(filename, end=' ' * 30 + '\r', flush=True)

        with open(filename, 'r', encoding='utf8') as r:
            expected = r.read()

        cleaned = hebrew.remove_niqqud(expected)
        actual = fetch_dicta_count_ambiguity(cleaned)
        c.update(actual)

    with open('count_ambiguity.txt', 'w', encoding='utf8') as f:
        print(c, file=f)


if __name__ == '__main__':
    # diacritize_all('NakdimonValidation')
    # diacritize_all('Nakdan')
    # print(diacritize("Nakdimon", 'tmp_expected.txt'))
    count_all_ambiguity()
