from __future__ import annotations
import collections
import logging
from pathlib import Path

import external_apis
import utils

import hebrew


def diacritize_all(system: str, basepath: str) -> None:
    diacritizer = external_apis.SYSTEMS[system]

    def diacritize_this(filename: str) -> None:
        infile = Path(filename)
        outfile = Path(filename.replace('expected', system))
        if outfile.exists():
            logging.info(f'{outfile} already exists, skipping')
            return
        outfile.parent.mkdir(parents=True, exist_ok=True)
        with open(infile, 'r', encoding='utf8') as f:
            expected = f.read()
        cleaned = hebrew.remove_niqqud(expected)
        actual = diacritizer(cleaned)
        logging.debug(f'Writing {outfile}')
        with open(outfile, 'w', encoding='utf8') as f:
            f.write(actual)
        logging.info(f'Diacritized: {outfile}')

    logging.info(f'Iterating over {basepath}')
    for filename in utils.iterate_files([basepath]):
        logging.info(f'Diacritizing {filename}')
        try:
            diacritize_this(filename)
        except external_apis.DottingError:
            logging.warning("Failed to dot")


def count_all_ambiguity(basepath: str) -> None:
    c = collections.Counter()
    for filename in utils.iterate_files([basepath]):
        print(filename, end=' ' * 30 + '\r', flush=True)

        with open(filename, 'r', encoding='utf8') as r:
            expected = r.read()

        cleaned = hebrew.remove_niqqud(expected)
        actual = external_apis.fetch_dicta_count_ambiguity(cleaned)
        c.update(actual)

    with open('count_ambiguity.txt', 'w', encoding='utf8') as f:
        print(c, file=f)


def main(system: str, test_set: str) -> None:
    if 'Maj' in system:
        external_apis.SYSTEMS.update(external_apis.prepare_majority())
    diacritize_all(system, f'{test_set}/expected')
