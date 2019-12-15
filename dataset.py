import tensorflow as tf
import numpy as np

import hebrew


class CharacterTable:
    START_TOKEN = '@'
    PAD_TOKEN = '^'

    def __init__(self, chars):
        self.chars = [self.PAD_TOKEN, self.START_TOKEN, ''] + list(sorted(set(chars)))
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))

    def __len__(self):
        return len(self.chars)

    def to_ids(self, css):
        return [
            [self.char_indices[self.START_TOKEN]] + [self.char_indices[c] for c in cs] for cs in css
        ]

    def __repr__(self):
        return repr(self.chars)


def from_categorical(t):
    return np.argmax(t, axis=-1)


class Data:
    input_texts: np.ndarray = None
    dagesh_texts: np.ndarray = None
    sin_texts: np.ndarray = None
    niqqud_texts: np.ndarray = None

    input_validation: np.ndarray = None
    dagesh_validation: np.ndarray = None
    sin_validation: np.ndarray = None
    niqqud_validation: np.ndarray = None

    letters_table: CharacterTable = None
    dagesh_table: CharacterTable = None
    sin_table: CharacterTable = None
    niqqud_table: CharacterTable = None

    def merge(self, ts, ds=None, ss=None, ns=None):
        default = [[''] * len(ts[0]) for x in range(len(ts))]
        texts = [[self.letters_table.indices_char[x] for x in line] for line in ts]
        dageshs = [[self.dagesh_table.indices_char[x] for x in xs] for xs in ds] if ds is not None else default
        sins = [[self.sin_table.indices_char[x] for x in xs] for xs in ss] if ss is not None else default
        niqquds = [[self.niqqud_table.indices_char[x] for x in xs] for xs in ns] if ns is not None else default
        assert len(texts) == len(niqquds)
        res = []
        for i in range(len(texts)):
            sentence = []
            for c, d, s, n in zip(texts[i], dageshs[i], sins[i], niqquds[i]):
                if c == self.letters_table.START_TOKEN:
                    continue
                if c == self.letters_table.PAD_TOKEN:
                    break
                sentence.append(c)
                sentence.append(d)
                sentence.append(s)
                sentence.append(n)
            res.append(''.join(sentence))
        return res


def load_file(batch_size, validation_rate, filenames, maxlen=100, shuffle=True) -> Data:
    heb_items = []
    for filename in filenames:
        with open(filename, encoding='utf-8') as f:
            text = ' '.join(f.read().split())
            heb_items.extend(hebrew.iterate_dotted_text(text))

    splitted_lines = list(hebrew.split_by_length(heb_items, maxlen))
    if shuffle:
        np.random.shuffle(splitted_lines)

    input_texts, dagesh_texts, sin_texts, niqqud_texts = zip(*(zip(*line) for line in splitted_lines))
    for x in input_texts:
        assert len(x) <= maxlen, len(x)
    data = Data()

    data.letters_table = CharacterTable(hebrew.HEBREW_LETTERS + ''.join(x for xs in input_texts for x in xs))
    data.dagesh_table = CharacterTable(hebrew.DAGESH)
    data.sin_table = CharacterTable(hebrew.NIQQUD_SIN)
    data.niqqud_table = CharacterTable(hebrew.NIQQUD)

    m = len(input_texts) // batch_size * batch_size

    def pad(table, css):
        return tf.keras.preprocessing.sequence.pad_sequences(table.to_ids(css[:m]), maxlen=maxlen,
                                                             dtype='int32', padding='post', truncating='post', value=0)

    input_texts = pad(data.letters_table, input_texts)

    dagesh_texts = pad(data.dagesh_table, dagesh_texts)
    sin_texts = pad(data.sin_table, sin_texts)
    niqqud_texts = pad(data.niqqud_table, niqqud_texts)

    data.letters_size = len(data.letters_table)

    v = int(m*(1-validation_rate))
    data.input_texts, data.input_validation = input_texts[:v], input_texts[v:]
    data.dagesh_texts, data.dagesh_validation = dagesh_texts[:v], dagesh_texts[v:]
    data.sin_texts, data.sin_validation = sin_texts[:v], sin_texts[v:]
    data.niqqud_texts, data.niqqud_validation = niqqud_texts[:v], niqqud_texts[v:]
    return data
