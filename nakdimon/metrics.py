from typing import Iterator, Optional
from pathlib import Path
from dataclasses import dataclass
import numpy as np

import external_apis
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
        return hebrew.tokenize(self.hebrew_items(), strip_nonhebrew=False)

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


def system_path_from_expected(path_to_expected: Path, system: str) -> Path:
    return Path(str(path_to_expected).replace('expected', system))


def read_document(system: str, path: Path) -> Document:
    return Document(path.name, system, ' '.join(path.read_text(encoding='utf8').strip().split()))


def read_document_pack(path_to_expected: Path, systems: list[str]) -> DocumentPack:
    return DocumentPack(path_to_expected.parent.name, path_to_expected.name,
                        {system: read_document(system, system_path_from_expected(path_to_expected, system))
                         for system in systems})

class Stats:
    def __init__(self, basepath: Path, vocabulary: Optional[external_apis.MajorityDiacritizer]):
        self.basepath = basepath
        self.vocabulary = vocabulary

    def iter_documents(self, systems: list[str]) -> Iterator[DocumentPack]:
        for folder in self.basepath.iterdir():
            for path_to_expected in folder.iterdir():
                yield read_document_pack(path_to_expected, systems)

    def iter_documents_by_folder(self, systems: list[str]) -> Iterator[list[DocumentPack]]:
        for folder in self.basepath.iterdir():
            yield [read_document_pack(path_to_expected, systems) for path_to_expected in folder.iterdir()]

    def collect_failed_tokens(self, doc_pack: DocumentPack, context: int):
        tokens_of = {system: doc_pack[system].tokens() for system in doc_pack.docs}
        for i in range(len(tokens_of['expected'])):
            res = {system: str(tokens_of[system][i]) for system in doc_pack.docs}
            if len(set(res.values())) > 1:
                pre_nonhebrew, _, post_nonhebrew = tokens_of['expected'][i].split_on_hebrew()
                pre = ' '.join(token.to_text() for token in tokens_of['expected'][i-context:i]) + ' ' + pre_nonhebrew
                post = post_nonhebrew + " " + ' '.join(token.to_text() for token in tokens_of['expected'][i+1:i+context+1])
                res = {system: tokens_of[system][i].split_on_hebrew()[1].to_text() for system in doc_pack.docs}
                yield (pre, res, post)

    def metric_cha(self, doc_pack: DocumentPack) -> float:
        """
        Calculate character-level agreement between actual and expected.
        """
        return self.mean_equal((x, y) for x, y in zip(doc_pack.actual.hebrew_items(), doc_pack.expected.hebrew_items())
                               if hebrew.can_any(x.letter))

    def metric_dec(self, doc_pack: DocumentPack) -> float:
        """
        Calculate nontrivial-decision agreement between actual and expected.
        """
        actual_hebrew = doc_pack.actual.hebrew_items()
        expected_hebrew = doc_pack.expected.hebrew_items()

        return self.mean_equal(
           ((x.niqqud, y.niqqud) for x, y in zip(actual_hebrew, expected_hebrew)
            if hebrew.can_niqqud(x.letter)),

           ((x.dagesh, y.dagesh) for x, y in zip(actual_hebrew, expected_hebrew)
            if hebrew.can_dagesh(x.letter)),

           ((x.sin, y.sin) for x, y in zip(actual_hebrew, expected_hebrew)
            if hebrew.can_sin(x.letter)),
        )

    def is_oov(self, word: str) -> bool:
        if self.vocabulary is None:
            return True
        return self.vocabulary.is_oov(word)

    def metric_wor(self, doc_pack: DocumentPack, oov=False) -> float:
        """
        Calculate token-level agreement between actual and expected,
        for tokens containing at least 2 Hebrew letters.
        """
        return self.mean_equal((x, y) for x, y in zip(doc_pack.actual.tokens(), doc_pack.expected.tokens())
                               if x.is_hebrew() and (not oov or self.is_oov(str(x))))

    def metric_voc(self, doc_pack: DocumentPack, oov=False) -> float:
        """
        Calculate token-level agreement over vocalization, between actual and expected,
        for tokens containing at least 2 Hebrew letters.
        """
        return self.mean_equal((x, y) for x, y in zip(doc_pack.actual.vocalized_tokens(), doc_pack.expected.vocalized_tokens())
                               if x.is_hebrew() and (not oov or self.is_oov(str(x))))

    def mean_equal(self, *pair_iterables) -> float:
        total = 0
        acc = 0
        for pair_iterable in pair_iterables:
            pair_iterable = list(pair_iterable)
            total += len(pair_iterable)
            acc += sum(x == y for x, y in pair_iterable)
        if not total:
            return 0
        return acc / total

    def all_metrics(self, doc_pack: DocumentPack) -> dict[str, float]:
        return {
            'dec': self.metric_dec(doc_pack),
            'cha': self.metric_cha(doc_pack),
            'wor': self.metric_wor(doc_pack),
            'voc': self.metric_voc(doc_pack),
            'wor_oov': self.metric_wor(doc_pack, oov=True),
            'voc_oov': self.metric_voc(doc_pack, oov=True),
        }

    def metricwise_mean(self, iterable) -> dict[str, float]:
        items = list(iterable)
        keys = items[0].keys()
        return {
            key: np.mean([item[key] for item in items])
            for key in keys
        }

    def macro_average(self, system: str) -> dict[str, float]:
        return self.metricwise_mean(
            self.metricwise_mean(self.all_metrics(doc_pack) for doc_pack in folder_packs)
            for folder_packs in self.iter_documents_by_folder(['expected', system])
        )

    def micro_average(self, system: str, vocabulary) -> dict[str, float]:
        return self.metricwise_mean(self.all_metrics(doc_pack) for doc_pack in self.iter_documents(['expected', system]))

    def all_failed(self) -> None:
        for doc_pack in self.iter_documents(['expected', 'Morfix', 'Dicta', 'Nakdimon']):
            for pre, ngrams, post in self.collect_failed_tokens(doc_pack, context=3):
                res = "|".join(ngrams.values())
                print(f'{doc_pack.source}|{doc_pack.name}| {pre}|{res}|{post} |')


def format_latex(system, results) -> None:
    print(r'{system} & {dec:.2%} & {cha:.2%} & {wor:.2%} & {voc:.2%} & {wor_oov:.2%} & {voc_oov:.2%} \\'.format(system=system, **results)
          .replace('%', ''))


def adapt_morfix(expected_filename: str) -> None:
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
        e = expected[i+1].letter
        a = actual[j+1].letter
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
            i += 1
        i += 1
    fixed_actual_filename = Path(expected_filename.replace('expected', 'Morfix'))
    fixed_actual_filename.parent.mkdir(parents=True, exist_ok=True)
    with open(fixed_actual_filename, 'w', encoding='utf8') as f:
        print(hebrew.items_to_text(fixed_actual), file=f)


def main(*, test_set: str, systems: list[str]) -> None:
    external_apis.SYSTEMS.update(external_apis.prepare_majority())

    maj = external_apis.MAJ_ALL_WITH_DICTA
    if test_set == 'tests/dicta':
        # Remove systems that depend on the dicta test set for their training
        maj = external_apis.MAJ_ALL_NO_DICTA
        del external_apis.SYSTEMS[external_apis.MAJ_ALL_WITH_DICTA]
        del external_apis.SYSTEMS['Nakdimon']
    elif test_set == 'tests/new':
        del external_apis.SYSTEMS[external_apis.MAJ_ALL_NO_DICTA]
    else:
        assert False
    stats = Stats(
        basepath=Path(f'{test_set}/expected'),
        vocabulary=external_apis.SYSTEMS[maj]
    )
    for system in systems:
        results = stats.macro_average(system)
        format_latex(system, results)
