import os
from keras import preprocessing
import numpy as np

import hebrew


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


def print_tables():
    print(letters_table.chars)
    print(niqqud_table.chars)
    print(dagesh_table.chars)
    print(sin_table.chars)

    
def from_categorical(t):
    return np.argmax(t, axis=-1)


class Data:
    input_texts: np.ndarray = None

    normalized_texts: np.ndarray = None
    dagesh_texts: np.ndarray = None
    sin_texts: np.ndarray = None
    niqqud_texts: np.ndarray = None

    normalized_validation: np.ndarray = None
    dagesh_validation: np.ndarray = None
    sin_validation: np.ndarray = None
    niqqud_validation: np.ndarray = None

    def merge(self, ts, ds=None, ss=None, ns=None):
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


def iterate_files(base_paths):
    for name in base_paths:
        if not os.path.isdir(name):
            yield name
            continue
        for root, dirs, files in os.walk(name):
            for fname in files:
                path = os.path.join(root, fname)
                yield path


def load_file(base_paths, batch_size, validation_rate, maxlen=100, shuffle=True) -> Data:
    heb_items = []
    for path in iterate_files(base_paths):
        with open(path, encoding='utf-8') as f:
            text = ' '.join(f.read().split())
            heb_items.extend(hebrew.iterate_dotted_text(text))

    splitted_lines = list(hebrew.split_by_length(heb_items, maxlen))
    if shuffle:
        np.random.shuffle(splitted_lines)

    input_texts, dagesh_texts, sin_texts, niqqud_texts = zip(*(zip(*line) for line in splitted_lines))

    m = len(input_texts) // batch_size * batch_size

    def pad(ords):
        return preprocessing.sequence.pad_sequences(ords, maxlen=maxlen, dtype='int32', padding='post', truncating='post', value=0)

    data = Data()
    data.input_texts = input_texts

    normalized_texts = pad(letters_table.to_ids(input_texts[:m]))

    dagesh_texts = pad(dagesh_table.to_ids(dagesh_texts[:m]))
    sin_texts = pad(sin_table.to_ids(sin_texts[:m]))
    niqqud_texts = pad(niqqud_table.to_ids(niqqud_texts[:m]))

    v = int(m*(1-validation_rate))
    data.normalized_texts, data.normalized_validation = normalized_texts[:v], normalized_texts[v:]
    data.dagesh_texts, data.dagesh_validation = dagesh_texts[:v], dagesh_texts[v:]
    data.sin_texts, data.sin_validation = sin_texts[:v], sin_texts[v:]
    data.niqqud_texts, data.niqqud_validation = niqqud_texts[:v], niqqud_texts[v:]
    return data


if __name__ == '__main__':
    modern = ['ali_baba.txt', 'uriel_ofek']
    filenames = [os.path.join('texts', f) for f in modern]
    data = load_file(filenames, 32, 0.01, maxlen=60, shuffle=True)
    print(data.normalized_texts[0])
