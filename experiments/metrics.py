from typing import Tuple, List
from pathlib import Path

import numpy as np

import hebrew


basepath = Path('tests/validation/expected')


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


def is_hebrew(token):
    return len([c for c in token.items if c.letter in hebrew.HEBREW_LETTERS]) > 1


def metric_wor(actual: str, expected: str, *args, **kwargs) -> float:
    """
    Calculate token-level agreement between actual and expected, for tokens containing at least 2 Hebrew letters.
    """
    actual_hebrew, expected_hebrew = get_items(actual, expected, *args, **kwargs)
    actual_tokens = hebrew.tokenize(actual_hebrew)
    expected_tokens = hebrew.tokenize(expected_hebrew)

    return mean_equal((x, y) for x, y in zip(actual_tokens, expected_tokens)
                      if is_hebrew(x))


def token_to_text(token: hebrew.Token) -> str:
    return str(token).replace(hebrew.RAFE, '')


def print_different_words(actual: str, expected: str, *args, **kwargs):
    actual_hebrew, expected_hebrew = get_items(actual, expected, *args, **kwargs)
    actual_tokens = hebrew.tokenize(actual_hebrew)
    expected_tokens = hebrew.tokenize(expected_hebrew)

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
    diff = get_diff(repr(''.join(c.letter for c in actual_hebrew)),
                    repr(''.join(c.letter for c in expected_hebrew)))
    assert not diff, diff
    return actual_hebrew, expected_hebrew


def split_to_sentences(text):
    return [sent + '.' for sent in text.split('. ') if len(hebrew.remove_niqqud(sent)) > 15]


def clean_read(filename):
    with open(filename, encoding='utf8') as f:
        return cleanup(f.read())


def all_diffs_for_files(expected_filename, system1, system2):
    expected_sentences = split_to_sentences(clean_read(expected_filename))
    actual_sentences1 = split_to_sentences(clean_read(expected_filename.replace('expected', system1)))
    actual_sentences2 = split_to_sentences(clean_read(expected_filename.replace('expected', system2)))
    assert len(expected_sentences) == len(actual_sentences1) == len(actual_sentences2)

    triples = [(e, a1, a2) for (e, a1, a2) in zip(expected_sentences, actual_sentences1, actual_sentences2)
               if metric_cha(a1, e) > 0.98 and metric_cha(a2, e) < 0.95]
    triples.sort(key=lambda e_a1_a2: metric_cha(e_a1_a2[2], e_a1_a2[0]))
    for (e, a1, a2) in triples[:20]:
        print(f"{system1}: {metric_cha(a1, e):.2%}; {system2}: {metric_cha(a2, e):.2%}")
        print('סבבה:', a1)
        print('מקור:', e)
        print('גרוע:', a2)
        print()


def all_diffs(system1, system2):
    for folder in basepath.iterdir():
        for file in folder.iterdir():
            all_diffs_for_files(str(file), system1, system2)


def all_metrics(actual, expected):
    return {
        'cha': metric_cha(actual, expected),
        'dec': metric_dec(actual, expected),
        'wor': metric_wor(actual, expected),
        'voc': metric_wor(actual, expected, vocalize=True)
    }


def cleanup(text):
    return ' '.join(text.strip().split())


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
        metricwise_mean(all_metrics_for_files(actual_path(file, sysname), file)
                        for file in folder.iterdir())
        for folder in basepath.iterdir()
    )


def micro_average(sysname):
    return metricwise_mean(
        all_metrics_for_files(actual_path(file, sysname), file)
        for folder in basepath.iterdir()
        for file in folder.iterdir()
    )


def expected_path(file, sysname):
    return str(file).replace("\\" + sysname + "\\", "\\expected\\")


def actual_path(file, sysname):
    return str(file).replace("\\expected\\", "\\" + sysname + "\\")


# TODO: concatenate and average

def breakdown(sysname):
    return {
        folder.name: metricwise_mean(all_metrics_for_files(actual_path(file, sysname), file)
                                     for file in folder.iterdir())
        for folder in basepath.iterdir()
    }


def format_latex(sysname, results):
    print('{sysname} & {cha:.2%}  & {dec:.2%} & {wor:.2%} & {voc:.2%} \\\\'.format(sysname=sysname, **results)
          .replace('%', ''))

def all_stats():
    SYSTEMS = [
        # "Nakdimon",
        # "Nakdan",
        # "Snopi",
        # "Nakdimon0"
        "Morfix"
        # "NakdimonNoDicta"
    ]
    for sysname in SYSTEMS:
        results = macro_average(sysname)
        format_latex(sysname, results)

    print()

    for sysname in SYSTEMS:
        results = micro_average(sysname)
        format_latex(sysname, results)

    print()

    for sysname in SYSTEMS:
        all_results = breakdown(sysname)
        for source, results in all_results.items():
            print(source, ",", ", ".join(str(x) for x in results.values()))


def adapt_morfix(expected_filename):
    with open(expected_filename, encoding='utf8') as f:
        expected = list(hebrew.iterate_dotted_text(f.read()))
    actual_filename = expected_filename.replace('expected', 'MorfixOriginal')
    print(actual_filename)
    with open(actual_filename, encoding='utf8') as f:
        actual = list(hebrew.iterate_dotted_text(f.read()))
    fixed_actual = []
    i = 0
    for j in range(len(actual)):
        fixed_actual.append(actual[j])
        if j + 1 >= len(actual):
            break
        # print(fixed_actual[-1].letter, end='', flush=True)
        e = expected[i+1].letter
        a = actual[j+1].letter
        # print(a, end='', flush=True)
        if e != a:
            assert actual[j + 1].letter == expected[i + 2].letter, hebrew.items_to_text(actual[j-15:j+15]) + ' != ' + hebrew.items_to_text(expected[i-15:i+15])
            if e == 'י' or e == 'א':
                fixed_actual.append(hebrew.HebrewItem(e, e, '', '', ''))
            elif e == 'ל':  # רמאללה -> רמאלה
                t = fixed_actual.pop()
                fixed_actual.append(hebrew.HebrewItem(e, e, '', '', ''))
                fixed_actual.append(t)
            elif e == 'ו':
                if actual[j].niqqud == hebrew.Niqqud.KAMATZ and actual[j].letter == 'ו':
                    add = ''
                elif actual[j].niqqud in [hebrew.Niqqud.REDUCED_KAMATZ, hebrew.Niqqud.KAMATZ, hebrew.Niqqud.HOLAM]:
                    add = hebrew.Niqqud.HOLAM
                    fixed_actual[-1] = actual[j]._replace(niqqud='')
                elif actual[j].niqqud in [hebrew.Niqqud.KUBUTZ]:
                    add = hebrew.Niqqud.SHURUK
                    fixed_actual[-1] = actual[j]._replace(niqqud='')
                else:
                    add = ''
                fixed_actual.append(hebrew.HebrewItem(e, e, '', '', add))
            else:
                assert False, hebrew.items_to_text(actual[j-15:j+15])
            # print(fixed_actual[-1].letter, end='', flush=True)
            i += 1
        i += 1
    fixed_actual_filename = Path(expected_filename.replace('expected', 'Morfix'))
    fixed_actual_filename.parent.mkdir(parents=True, exist_ok=True)
    with open(fixed_actual_filename, 'w', encoding='utf8') as f:
        print(hebrew.items_to_text(fixed_actual), file=f)


if __name__ == '__main__':
    # for i in range(1, 23):
    #     adapt_morfix(f'test_dicta/expected/dicta/{i}.txt')
    # all_diffs('Nakdan', 'NakdimonValidation')
    # all_diffs('NakdimonValidation', 'Nakdan')
    all_stats()
