import re
from typing import Tuple, List
from pathlib import Path

import numpy as np

import hebrew


basepath = Path('test/expected')


def metric_cha(actual: str, expected: str, *args, **kwargs) -> float:
    """
    Calculate character-level agreement between actual and expected.
    """
    actual_hebrew, expected_hebrew = get_items(actual, expected, *args, **kwargs)
    return mean_equal((x, y) for x, y in zip(actual_hebrew, expected_hebrew)
                      if hebrew.can_any(x.letter))


def metric_dec(actual: str, expected: str, *args, **kwargs) -> float:
    """
    Calculate nontrivial-decision agreement between actual and expected.
    """
    actual_hebrew, expected_hebrew = get_items(actual, expected, *args, **kwargs)

    return mean_equal(
       ((x.niqqud, y.niqqud) for x, y in zip(actual_hebrew, expected_hebrew)
        if hebrew.can_niqqud(x.letter)),

       ((x.dagesh, y.dagesh) for x, y in zip(actual_hebrew, expected_hebrew)
        if hebrew.can_dagesh(x.letter)),

       ((x.sin, y.sin) for x, y in zip(actual_hebrew, expected_hebrew)
        if hebrew.can_sin(x.letter)),
    )


def metric_wor(actual: str, expected: str, *args, **kwargs) -> float:
    """
    Calculate token-level agreement between actual and expected, for tokens containing at least 2 Hebrew letters.
    """
    actual_hebrew, expected_hebrew = get_items(actual, expected, *args, **kwargs)
    actual_tokens = hebrew.tokenize(actual_hebrew)
    expected_tokens = hebrew.tokenize(expected_hebrew)

    def is_hebrew(token):
        return len([c for c in token.items if c.letter in hebrew.HEBREW_LETTERS]) > 1

    return mean_equal((x, y) for x, y in zip(actual_tokens, expected_tokens)
                      if is_hebrew(x))


def token_to_text(token: hebrew.Token) -> str:
    return str(token).replace(hebrew.RAFE, '')


def print_different_words(actual: str, expected: str, *args, **kwargs):
    actual_hebrew, expected_hebrew = get_items(actual, expected, *args, **kwargs)
    actual_tokens = hebrew.tokenize(actual_hebrew)
    expected_tokens = hebrew.tokenize(expected_hebrew)

    def is_hebrew(token):
        return len([c for c in token.items if c.letter in hebrew.HEBREW_LETTERS]) > 1

    diff = [(token_to_text(x), token_to_text(y)) for x, y in zip(actual_tokens, expected_tokens)
            if is_hebrew(x) and x != y]

    for x, y in diff:
        print(x, y)


def mean_equal(*pair_iterables):
    total = 0
    acc = 0
    for pair_iterable in pair_iterables:
        pair_iterable = list(pair_iterable)
        total += len(pair_iterable)
        acc += sum(x == y for x, y in pair_iterable)
    return acc / total


def get_diff(actual, expected):
    for i, (a, e) in enumerate(zip(actual, expected)):
        if a != e:
            return f'\n{actual[i-15:i+15]}\n!=\n{expected[i-15:i+15]}'
    return ''


def get_items(actual: str, expected: str, vocalize=False) -> Tuple[List[hebrew.HebrewItem], List[hebrew.HebrewItem]]:
    expected_hebrew = list(hebrew.iterate_dotted_text(expected))
    actual_hebrew = list(hebrew.iterate_dotted_text(actual))
    if vocalize:
        expected_hebrew = [x.vocalize() for x in expected_hebrew]
        actual_hebrew = [x.vocalize() for x in actual_hebrew]
    diff = get_diff(''.join(c.letter for c in actual_hebrew),
                    ''.join(c.letter for c in expected_hebrew))
    assert not diff, diff
    return actual_hebrew, expected_hebrew


def all_metrics(actual, expected):
    return {
        'cha': metric_cha(actual, expected),
        'dec': metric_dec(actual, expected),
        'wor': metric_wor(actual, expected),
        'voc': metric_wor(actual, expected, vocalize=True)
    }


def cleanup(text):
    return re.sub(r'\s+', ' ', text.strip(), re.MULTILINE | re.DOTALL)


def calculate_metrics(predict):
    for file in Path('./validation/expected/modern/').glob('*'):
        print(file)
        with open(file, encoding='utf8') as f:
            expected = cleanup(f.read())
        actual = cleanup(predict(hebrew.remove_niqqud(expected)))
        yield all_metrics(actual, expected)


def all_metrics_for_files(actual_filename, expected_filename):
    with open(expected_filename, encoding='utf8') as f:
        expected = cleanup(f.read())

    with open(actual_filename, encoding='utf8') as f:
        actual = cleanup(f.read())
    try:
        return all_metrics(actual, expected)
    except AssertionError as ex:
        raise RuntimeError(actual_filename) from ex


def metricwise_mean(iterable):
    items = list(iterable)
    keys = items[0].keys()
    return {
        key: np.mean([item[key] for item in items])
        for key in keys
    }


def macro_average(sysname):
    return metricwise_mean(
        metricwise_mean(all_metrics_for_files(file, str(file).replace("expected", sysname))
                        for file in folder.iterdir())
        for folder in basepath.iterdir()
    )


def micro_average(sysname):
    return metricwise_mean(
        all_metrics_for_files(file, str(file).replace("expected", sysname))
        for folder in basepath.iterdir()
        for file in folder.iterdir()
    )


def breakdown(sysname):
    return {
        folder.name: metricwise_mean(all_metrics_for_files(file, str(file).replace("expected", sysname))
                                     for file in folder.iterdir())
        for folder in basepath.iterdir()
    }


def format_latex(sysname, results):
    print('{sysname} & {cha:.2%}  & {dec:.2%} & {wor:.2%} & {voc:.2%} \\\\'.format(sysname=sysname, **results)
          .replace('%', ''))


if __name__ == '__main__':
    SYSTEMS = [
        # "Nakdimon",
        # "Nakdan",
        # "Snopi",
        # "Nakdimon0"
    ]
    for sysname in SYSTEMS:
        results = macro_average(sysname)
        format_latex(sysname, results)

    print()

    for sysname in SYSTEMS:
        all_results = breakdown(sysname)
        for source, results in all_results.items():
            print(source, ",", ", ".join(str(x) for x in results.values()))
