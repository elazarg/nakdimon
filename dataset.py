from typing import Tuple, List

import os
from tensorflow.keras import preprocessing
import tensorflow as tf
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


def merge(ts, ds=None, ss=None, ns=None):
    default = [[''] * len(ts[0]) for x in range(len(ts))]
    texts = [[letters_table.indices_char[x] for x in line] for line in ts]
    dageshs = [[dagesh_table.indices_char[x] for x in xs] for xs in ds] if ds is not None else default
    sins = [[sin_table.indices_char[x] for x in xs] for xs in ss] if ss is not None else default
    niqquds = [[niqqud_table.indices_char[x] for x in xs] for xs in ns] if ns is not None else default
    assert len(texts) == len(niqquds)
    res = []
    for i in range(len(texts)):
        sentence = []
        for c, d, s, n in zip(texts[i], dageshs[i], sins[i], niqquds[i]):
            if c == letters_table.PAD_TOKEN:
                break
            sentence.append(c)
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
        self.kind = np.concatenate([x.kind for x in others])
        return self

    def shapes(self):
        return self.text.shape, self.normalized.shape, self.dagesh.shape, self.sin.shape, self.niqqud.shape, self.kind.shape

    @staticmethod
    def from_text(heb_items, maxlen: int) -> 'Data':
        self = Data()
        text, dagesh, sin, niqqud = zip(*(zip(*line) for line in hebrew.split_by_length(heb_items, maxlen)))

        def pad(ords, dtype='int32', value=0):
            return preprocessing.sequence.pad_sequences(ords, maxlen=maxlen,
                        dtype=dtype, padding='post', truncating='post', value=value)

        self.normalized = pad(letters_table.to_ids(text))
        self.dagesh = pad(dagesh_table.to_ids(dagesh))
        self.sin = pad(sin_table.to_ids(sin))
        self.niqqud = pad(niqqud_table.to_ids(niqqud))
        self.text = pad(text, dtype='<U1', value=0)
        return self

    def add_kind(self, path):
        dirname = path.replace(os.path.sep, '/').split('/')[1]
        self.kind = np.full(len(self), KINDS.index(dirname))

    def __len__(self):
        return self.normalized.shape[0]  # len(input_texts) // batch_size * batch_size

    def split_validation(self, validation_rate):
        indices = np.random.permutation(len(self))
        v = int(len(self) * (1 - validation_rate))
        valid_idx, test_idx = indices[:v], indices[v:]
        valid = Data()
        self.text, valid.text = self.text[valid_idx], self.text[test_idx]
        self.normalized, valid.normalized = self.normalized[valid_idx], self.normalized[test_idx]
        self.dagesh, valid.dagesh = self.dagesh[valid_idx], self.dagesh[test_idx]
        self.sin, valid.sin = self.sin[valid_idx], self.sin[test_idx]
        self.niqqud, valid.niqqud = self.niqqud[valid_idx], self.niqqud[test_idx]
        self.kind, valid.kind = self.kind[valid_idx], self.kind[test_idx]
        return valid

    def print_stats(self):
        print(self.shapes())


def load_file(path: str, maxlen: int) -> Data:
    with open(path, encoding='utf-8') as f:
        text = ' '.join(f.read().split())
    res = Data.from_text(hebrew.iterate_dotted_text(text), maxlen)
    res.add_kind(path)
    return res


def load_data(base_paths: List[str], validation_rate: float, maxlen: int) -> Tuple[Data, Data]:
    corpus = [load_file(path, maxlen) for path in utils.iterate_files(base_paths)]
    result = Data.concatenate(corpus)
    validation = result.split_validation(validation_rate)
    return result, validation


class CircularLearningRate(tf.keras.callbacks.Callback):
    def __init__(self, min_lr_1, max_lr, min_lr_2):
        super().__init__()
        self.min_lr_1 = min_lr_1
        self.max_lr = max_lr
        self.min_lr_2 = min_lr_2

    def set_dataset(self, data, batch_size):
        self.mid = len(data) / batch_size / 2

    def on_train_batch_end(self, batch, logs=None):
        if batch < self.mid:
            lb = self.min_lr_1
            way = self.mid - batch
        else:
            lb = self.min_lr_2
            way = batch - self.mid
        lr = self.max_lr - way / self.mid * (self.max_lr - lb)
        tf.keras.backend.set_value(self.model.optimizer.lr, lr)


if __name__ == '__main__':
    data, valid = load_data(['texts/modern'], 0.1, maxlen=64)
    data.print_stats()
    print(data.kind[0])
    valid.print_stats()
