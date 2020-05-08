from typing import Tuple, List

import os
import numpy as np

import hebrew
import utils


class CharacterTable:
    PAD_TOKEN = ''
    HIDDEN_TOKEN = ''

    def __init__(self, chars):
        # make sure to be consistent with JS
        self.chars = [CharacterTable.PAD_TOKEN, CharacterTable.HIDDEN_TOKEN] + chars
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))

    def __len__(self):
        return len(self.chars)

    def to_ids(self, css):
        return [
            [self.char_indices[c] for c in cs] for cs in css
        ]

    def __repr__(self):
        return repr(self.chars)


letters_table = CharacterTable(hebrew.SPECIAL_TOKENS + hebrew.VALID_LETTERS)
dagesh_table = CharacterTable(hebrew.DAGESH)
sin_table = CharacterTable(hebrew.NIQQUD_SIN)
niqqud_table = CharacterTable(hebrew.NIQQUD)
KINDS = ('biblical', 'rabanit', 'poetry', 'pre_modern', 'modern', 'garbage')


def print_tables():
    print(letters_table.chars)
    print(niqqud_table.chars)
    print(dagesh_table.chars)
    print(sin_table.chars)

    
def from_categorical(t):
    return np.argmax(t, axis=-1)


def merge(texts, ts, ds, ss, ns):
    normalizeds = [[letters_table.indices_char[x] for x in line] for line in ts]
    dageshs = [[dagesh_table.indices_char[x] for x in xs] for xs in ds]
    sins = [[sin_table.indices_char[x] for x in xs] for xs in ss]
    niqquds = [[niqqud_table.indices_char[x] for x in xs] for xs in ns]
    assert len(normalizeds) == len(niqquds)
    res = []
    for i in range(len(texts)):
        sentence = []
        for t, c, d, s, n in zip(texts[i], normalizeds[i], dageshs[i], sins[i], niqquds[i]):
            if c == letters_table.PAD_TOKEN:
                break
            sentence.append(t)
            sentence.append(d)
            sentence.append(s)
            sentence.append(n)
        res.append(''.join(sentence))
    return res


class Data:
    text: np.ndarray = None
    normalized: np.ndarray = None
    dagesh: np.ndarray = None
    sin: np.ndarray = None
    niqqud: np.ndarray = None
    kind: np.ndarray = None

    @staticmethod
    def concatenate(others):
        self = Data()
        self.text = np.concatenate([x.text for x in others])
        self.normalized = np.concatenate([x.normalized for x in others])
        self.dagesh = np.concatenate([x.dagesh for x in others])
        self.sin = np.concatenate([x.sin for x in others])
        self.niqqud = np.concatenate([x.niqqud for x in others])
        # self.kind = np.concatenate([x.kind for x in others])
        self.shuffle()
        return self

    def shapes(self):
        return self.text.shape, self.normalized.shape, self.dagesh.shape, self.sin.shape, self.niqqud.shape #, self.kind.shape

    def shuffle(self):
        indices = np.random.permutation(len(self))
        self.text = self.text[indices]
        self.normalized = self.normalized[indices]
        self.dagesh = self.dagesh[indices]
        self.niqqud = self.niqqud[indices]
        self.sin = self.sin[indices]
        # self.kind = self.kind[indices]

    @staticmethod
    def from_text(heb_items, maxlen: int) -> 'Data':
        assert heb_items
        from tensorflow.keras import preprocessing
        self = Data()
        text, normalized, dagesh, sin, niqqud = zip(*(zip(*line) for line in hebrew.split_by_length(heb_items, maxlen)))

        def pad(ords, dtype='int32', value=0):
            return preprocessing.sequence.pad_sequences(ords, maxlen=maxlen,
                        dtype=dtype, padding='post', truncating='post', value=value)

        self.normalized = pad(letters_table.to_ids(normalized))
        self.dagesh = pad(dagesh_table.to_ids(dagesh))
        self.sin = pad(sin_table.to_ids(sin))
        self.niqqud = pad(niqqud_table.to_ids(niqqud))
        self.text = pad(text, dtype='<U1', value=0)
        return self

    def add_kind(self, path):
        base = path.replace(os.path.sep, '/').split('/')
        if len(base) > 1:
            dirname = base[1]
            self.kind = np.full(len(self), KINDS.index(dirname))

    def __len__(self):
        return self.normalized.shape[0]

    def print_stats(self):
        print(self.shapes())


def load_file(path: str, maxlen: int) -> Data:
    with open(path, encoding='utf-8') as f:
        text = ' '.join(f.read().split())
    res = Data.from_text(hebrew.iterate_dotted_text(text), maxlen)
    # res.add_kind(path)
    return res


def read_corpus(base_paths, maxlen):
    return [load_file(path, maxlen) for path in utils.iterate_files(base_paths)]


def load_data(base_paths: List[str], validation_rate: float, maxlen: int) -> Tuple[Data, Data]:
    corpus = read_corpus(base_paths, maxlen)
    np.random.shuffle(corpus)
    # result = Data.concatenate(corpus)
    # validation = result.split_validation(validation_rate)

    size = sum(len(x) for x in corpus)
    validation_size = size * validation_rate
    validation = []
    total_size = 0
    while total_size < validation_size:
        if abs(total_size - validation_size) < abs(total_size + len(corpus[-1]) - validation_size):
            break
        c = corpus.pop()
        total_size += len(c)
        validation.append(c)

    train = Data.concatenate(corpus)
    train.shuffle()
    return train, Data.concatenate(validation)


if __name__ == '__main__':
    data = Data.concatenate(read_corpus(['texts/modern/wiki/1.txt'], maxlen=64))
    data.print_stats()
    print(np.concatenate([data.normalized[:1], data.sin[:1]]))
    res = merge(data.normalized[:1], data.dagesh[:1], data.sin[:1], data.niqqud[:1])
    print(res)
