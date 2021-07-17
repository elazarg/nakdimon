import collections
from pathlib import Path

import utils

from external_apis import SYSTEMS, fetch_dicta_count_ambiguity
import hebrew

basepath = '../gender_dots/scraping/scrape_data/shortstoryproject'


def diacritize_all(sysname):
    diacritizer = SYSTEMS[sysname]

    def diacritize_this(filename):
        outfile = filename.replace('shortstoryproject', sysname)
        if Path(outfile).exists():
            return
        print(filename)
        Path(outfile).parent.mkdir(parents=True, exist_ok=True)
        with open(filename, 'r', encoding='utf8') as f:
            expected = f.read()
        cleaned = hebrew.remove_niqqud(expected)
        actual = diacritizer(cleaned)
        with open(outfile, 'w', encoding='utf8') as f:
            f.write(actual)

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=1) as executor:
        for filename in utils.iterate_files([basepath]):
            executor.submit(diacritize_this, filename)


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
    diacritize_all('Dicta')
    # print(diacritize("Nakdimon", 'tmp_expected.txt'))
    # count_all_ambiguity()
