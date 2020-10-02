from typing import Tuple, List

import os
import numpy as np

import hebrew
import utils


class CharacterTable:
    MASK_TOKEN = ''

    def __init__(self, chars):
        # make sure to be consistent with JS
        self.chars = [CharacterTable.MASK_TOKEN] + chars
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


def print_tables():
    print('const ALL_TOKENS =', letters_table.chars, end=';\n')
    print('const niqqud_array =', niqqud_table.chars, end=';\n')
    print('const dagesh_array =', dagesh_table.chars, end=';\n')
    print('const sin_array =', sin_table.chars, end=';\n')

    
def from_categorical(t):
    return np.argmax(t, axis=-1)


def merge(texts, tnss, nss, dss, sss):
    res = []
    for ts, tns, ns, ds, ss in zip(texts, tnss, nss, dss, sss):
        sentence = []
        for t, tn, n, d, s in zip(ts, tns, ns, ds, ss):
            if tn == 0:
                break
            sentence.append(t)
            if hebrew.can_dagesh(t):
                sentence.append(dagesh_table.indices_char[d].replace(hebrew.RAFE, ''))
            if hebrew.can_sin(t):
                sentence.append(sin_table.indices_char[s].replace(hebrew.RAFE, ''))
            if hebrew.can_niqqud(t):
                sentence.append(niqqud_table.indices_char[n].replace(hebrew.RAFE, ''))
        res.append(''.join(sentence))
    return res


class Data:
    text: np.ndarray = None
    normalized: np.ndarray = None
    dagesh: np.ndarray = None
    sin: np.ndarray = None
    niqqud: np.ndarray = None

    filenames: Tuple[str, ...] = ()

    @staticmethod
    def concatenate(others):
        self = Data()
        self.text = np.concatenate([x.text for x in others])
        self.normalized = np.concatenate([x.normalized for x in others])
        self.dagesh = np.concatenate([x.dagesh for x in others])
        self.sin = np.concatenate([x.sin for x in others])
        self.niqqud = np.concatenate([x.niqqud for x in others])
        return self

    def shapes(self):
        return self.text.shape, self.normalized.shape, self.dagesh.shape, self.sin.shape, self.niqqud.shape #, self.kind.shape

    def shuffle(self):
        utils.shuffle_in_unison(
            self.text,
            self.normalized,
            self.dagesh,
            self.niqqud,
            self.sin
        )

    @staticmethod
    def from_text(heb_items, maxlen: int) -> 'Data':
        assert heb_items
        self = Data()
        text, normalized, dagesh, sin, niqqud = zip(*(zip(*line) for line in hebrew.split_by_length(heb_items, maxlen)))

        def pad(ords, dtype='int32', value=0):
            return utils.pad_sequences(ords, maxlen=maxlen, dtype=dtype, value=value)

        self.normalized = pad(letters_table.to_ids(normalized))
        self.dagesh = pad(dagesh_table.to_ids(dagesh))
        self.sin = pad(sin_table.to_ids(sin))
        self.niqqud = pad(niqqud_table.to_ids(niqqud))
        self.text = pad(text, dtype='<U1', value=0)
        return self

    def __len__(self):
        return self.normalized.shape[0]

    def print_stats(self):
        print(self.shapes())


def read_corpora(base_paths):
    return [(filename, list(hebrew.iterate_file(filename))) for filename in utils.iterate_files(base_paths)]


def load_data(corpora, validation_rate: float, maxlen: int, shuffle=True) -> Tuple[Data, Data]:
    corpus = [(filename, Data.from_text(heb_items, maxlen)) for (filename, heb_items) in corpora]

    validation_data = None
    if validation_rate > 0:
        np.random.shuffle(corpus)
        size = sum(len(x) for _, x in corpus)
        validation_size = size * validation_rate
        validation = []
        validation_filenames: List[str] = []
        total_size = 0
        while total_size < validation_size:
            if abs(total_size - validation_size) < abs(total_size + len(corpus[-1]) - validation_size):
                break
            (filename, c) = corpus.pop()
            total_size += len(c)
            validation.append(c)
            validation_filenames.append(filename)
        validation_data = Data.concatenate(validation)
        validation_data.filenames = tuple(validation_filenames)

    train = Data.concatenate([c for (_, c) in corpus])
    if shuffle:
        train.shuffle()
    return train, validation_data


if __name__ == '__main__':
    # data = Data.concatenate([Data.from_text(x, maxlen=64) for x in read_corpora(['hebrew_diacritized/modern/wiki/1.txt'])])
    # data.print_stats()
    # print(np.concatenate([data.normalized[:1], data.sin[:1]]))
    # res = merge(data.text[:1], data.normalized[:1], data.niqqud[:1], data.dagesh[:1], data.sin[:1])
    # print(res)
    print_tables()
    print(letters_table.to_ids(["שלום"]))
