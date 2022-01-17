import typing
from pathlib import Path
from dataclasses import dataclass
import numpy as np

import hebrew


@dataclass(frozen=True)
class Document:
    name: str
    system: str
    text: str

    def sentences(self):
        return [sent + '.' for sent in self.text.split('. ') if len(hebrew.remove_niqqud(sent)) > 15]

    def hebrew_items(self) -> list[hebrew.HebrewItem]:
        return list(hebrew.iterate_dotted_text(self.text))

    def tokens(self) -> list[hebrew.Token]:
        return hebrew.tokenize(self.hebrew_items())

    def vocalized_tokens(self) -> list[hebrew.Token]:
        return [x.vocalize() for x in self.tokens()]


@dataclass(frozen=True)
class DocumentPack:
    source: str
    name: str
    docs: dict[str, Document]

    def __getitem__(self, item):
        return self.docs[item]

    @property
    def expected(self):
        return self.docs['expected']

    @property
    def actual(self):
        assert len(self.docs) == 2
        assert 'expected' in self.docs
        return self.docs[(set(self.docs.keys()) - {'expected'}).pop()]


def read_document(system: str, path: Path) -> Document:
    return Document(path.name, system, ' '.join(path.read_text(encoding='utf8').strip().split()))


def read_document_pack(path_to_expected: Path, *systems: str) -> DocumentPack:
    return DocumentPack(path_to_expected.parent.name, path_to_expected.name,
                        {system: read_document(system, system_path_from_expected(path_to_expected, system))
                         for system in systems})


def iter_documents(*systems) -> typing.Iterator[DocumentPack]:
    for folder in basepath.iterdir():
        for path_to_expected in folder.iterdir():
            yield read_document_pack(path_to_expected, *systems)


def iter_documents_by_folder(*systems) -> typing.Iterator[list[DocumentPack]]:
    for folder in basepath.iterdir():
        yield [read_document_pack(path_to_expected, *systems) for path_to_expected in folder.iterdir()]


def system_path_from_expected(path: Path, system: str) -> Path:
    return Path(str(path).replace('expected', system))


def collect_failed_tokens(doc_pack, context):
    tokens_of = {system: doc_pack[system].tokens() for system in doc_pack.docs}
    for i in range(len(tokens_of['expected'])):
        res = {system: str(tokens_of[system][i]) for system in doc_pack.docs}
        if len(set(res.values())) > 1:
            pre_nonhebrew, _, post_nonhebrew = tokens_of['expected'][i].split_on_hebrew()
            pre = ' '.join(token_to_text(x) for x in tokens_of['expected'][i-context:i]) + ' ' + pre_nonhebrew
            post = post_nonhebrew + " " + ' '.join(token_to_text(x) for x in tokens_of['expected'][i+1:i+context+1])
            res = {system: token_to_text(tokens_of[system][i].split_on_hebrew()[1]) for system in doc_pack.docs}
            yield (pre, res, post)


def metric_cha(doc_pack: DocumentPack) -> float:
    """
    Calculate character-level agreement between actual and expected.
    """
    return mean_equal((x, y) for x, y in zip(doc_pack.actual.hebrew_items(), doc_pack.expected.hebrew_items())
                      if hebrew.can_any(x.letter))


def metric_dec(doc_pack: DocumentPack) -> float:
    """
    Calculate nontrivial-decision agreement between actual and expected.
    """
    actual_hebrew = doc_pack.actual.hebrew_items()
    expected_hebrew = doc_pack.expected.hebrew_items()

    return mean_equal(
       ((x.niqqud, y.niqqud) for x, y in zip(actual_hebrew, expected_hebrew)
        if hebrew.can_niqqud(x.letter)),

       ((x.dagesh, y.dagesh) for x, y in zip(actual_hebrew, expected_hebrew)
        if hebrew.can_dagesh(x.letter)),

       ((x.sin, y.sin) for x, y in zip(actual_hebrew, expected_hebrew)
        if hebrew.can_sin(x.letter)),
    )


def is_hebrew(token: hebrew.Token) -> bool:
    return len([c for c in token.items if c.letter in hebrew.HEBREW_LETTERS]) > 1


def metric_wor(doc_pack: DocumentPack) -> float:
    """
    Calculate token-level agreement between actual and expected,
    for tokens containing at least 2 Hebrew letters.
    """
    return mean_equal((x, y) for x, y in zip(doc_pack.actual.tokens(), doc_pack.expected.tokens())
                      if is_hebrew(x))


def metric_voc(doc_pack: DocumentPack) -> float:
    """
    Calculate token-level agreement over vocalization, between actual and expected,
    for tokens containing at least 2 Hebrew letters.
    """
    return mean_equal((x, y) for x, y in zip(doc_pack.actual.vocalized_tokens(), doc_pack.expected.vocalized_tokens())
                      if is_hebrew(x))


def token_to_text(token: hebrew.Token) -> str:
    return str(token).replace(hebrew.RAFE, '')


def mean_equal(*pair_iterables):
    total = 0
    acc = 0
    for pair_iterable in pair_iterables:
        pair_iterable = list(pair_iterable)
        total += len(pair_iterable)
        acc += sum(x == y for x, y in pair_iterable)
    return acc / total


def all_diffs_for_files(doc_pack: DocumentPack, system1: str, system2: str) -> None:
    triples = [(e, a1, a2) for (e, a1, a2) in zip(doc_pack.expected.sentences(),
                                                  doc_pack[system1].sentences(),
                                                  doc_pack[system2].sentences())
               if metric_wor(a1, e) < 0.90 or metric_wor(a2, e) < 0.90]
    triples.sort(key=lambda e_a1_a2: metric_cha(e_a1_a2[2], e_a1_a2[0]))
    for (e, a1, a2) in triples[:20]:
        print(f"{system1}: {metric_wor(a1, e):.2%}; {system2}: {metric_wor(a2, e):.2%}")
        print('סבבה:', a1)
        print('מקור:', e)
        print('גרוע:', a2)
        print()


def all_metrics(doc_pack: DocumentPack):
    return {
        'dec': metric_dec(doc_pack),
        'cha': metric_cha(doc_pack),
        'wor': metric_wor(doc_pack),
        'voc': metric_voc(doc_pack)
    }


def metricwise_mean(iterable):
    items = list(iterable)
    keys = items[0].keys()
    return {
        key: np.mean([item[key] for item in items])
        for key in keys
    }


def macro_average(system):
    return metricwise_mean(
        metricwise_mean(all_metrics(doc_pack) for doc_pack in folder_packs)
        for folder_packs in iter_documents_by_folder('expected', system)
    )


def micro_average(system):
    return metricwise_mean(all_metrics(doc_pack) for doc_pack in iter_documents('expected', system))


def format_latex(system, results):
    print(r'{system} & {dec:.2%} & {cha:.2%} & {wor:.2%} & {voc:.2%} \\'.format(system=system, **results)
          .replace('%', ''))


def all_stats(*systems):
    for system in systems:
        results = macro_average(system)
        format_latex(system, results)
        results = micro_average(system)
        format_latex(system, results)
        ew = 1-results['wor']
        ev = 1 - results['voc']
        print(f'{(ew-ev)/ew:.2%}')
        print()


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


def all_failed():
    for doc_pack in iter_documents('expected', 'Morfix', 'Dicta', 'Nakdimon'):
        for pre, ngrams, post in collect_failed_tokens(doc_pack, context=3):
            res = "|".join(ngrams.values())
            print(f'{doc_pack.source}|{doc_pack.name}| {pre}|{res}|{post} |')


if __name__ == '__main__':
    basepath = Path('tests/dicta/expected')
    all_stats(
        'Snopi',
        'Morfix',
        'Dicta',
        'Nakdimon0',
        # 'Nakdimon',
    )
