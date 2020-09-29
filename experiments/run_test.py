from pathlib import Path

import utils

from external_apis import SYSTEMS
import hebrew

basepath = 'test/expected'


def diacritize(sysname, filename):
    with open(filename, 'r', encoding='utf8') as f:
        expected = f.read()
    cleaned = hebrew.remove_niqqud(expected)
    return SYSTEMS[sysname](cleaned)


def diacritize_all(sysname):
    for filename in utils.iterate_files([basepath]):
        # if filename in [r'test\nrg\6.txt', r'test\president\6.txt']:
        #    continue
        print(filename)

        actual = diacritize(sysname, filename)

        outfile = filename.replace('expected', sysname)
        Path(outfile).parent.mkdir(parents=True, exist_ok=True)

        with open(outfile, 'w', encoding='utf8') as f:
            f.write(actual)


if __name__ == '__main__':
    diacritize_all('Nakdimon')
    # print(diacritize("Nakdimon", 'tmp_expected.txt'))
