import collections
from pathlib import Path

import utils

from external_apis import SYSTEMS, fetch_dicta_count_ambiguity, DottingError
import hebrew


def diacritize_all(sysname, basepath):
    diacritizer = SYSTEMS[sysname]

    def diacritize_this(filename):
        infile = Path(filename)
        outfile = Path(filename.replace('expected', sysname))
        if outfile.exists():
            return
        print(outfile)
        outfile.parent.mkdir(parents=True, exist_ok=True)
        with open(infile, 'r', encoding='utf8') as f:
            expected = f.read()
        cleaned = hebrew.remove_niqqud(expected)
        actual = diacritizer(cleaned)
        with open(outfile, 'w', encoding='utf8') as f:
            f.write(actual)

    for filename in utils.iterate_files([basepath]):
        try:
            diacritize_this(filename)
        except DottingError:
            print("Failed to dot")


def count_all_ambiguity(basepath):
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
    diacritize_all('NakdimonFinalWithShortStory', 'tests/test/expected')
    # print(diacritize("Nakdimon", 'tmp_expected.txt'))
    # count_all_ambiguity()
