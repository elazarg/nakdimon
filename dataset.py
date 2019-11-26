import tensorflow as tf
import numpy as np


def is_text(c):
    return '\u05d0' <= c <= '\u05ea'


def is_niqqud(c):
    return '\u0591' <= c <= '\u05c7'


def iterate_dotted_text(line):
    n = len(line)
    line += '  '
    i = 0
    while i < n:
        dagesh = '_'
        niqqud = '_'
        sin = '_'
        c = line[i]
        i += 1
        if is_text(c):
            if line[i] == '\u05bc' and (c != 'ו' or is_niqqud(line[i+1])):
                dagesh = line[i]
                i += 1
            if line[i] in '\u05c1\u05c2':
                sin = line[i]
                i += 1
            if is_niqqud(line[i]):
                niqqud = line[i]
                i += 1
        yield (c, sin, dagesh, niqqud)


def unzip_dotted_text(line):
    return zip(*iterate_dotted_text(line))


def unzip_dotted_lines(lines, maxlen):
    ws, xs, ys, zs = [], [], [], []
    for line in lines:
        w, x, y, z = zip(*iterate_dotted_text(line))
        
        line = ''.join(w)
        n = line.rfind(' ')
        if n <= 0 or len(line) < maxlen:
            continue
        ws.append(w[:n])
        xs.append(x[:n])
        ys.append(y[:n])
        zs.append(z[:n])
    return ws, xs, ys, zs


class CharacterTable:
    START_TOKEN = '@'
    PAD_TOKEN = '^'

    def __init__(self, chars):
        self.chars = [self.PAD_TOKEN, self.START_TOKEN] + list(sorted(set(chars)))
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))

    def __len__(self):
        return len(self.chars)

    def to_ids(self, css):
        return [
            [self.char_indices[self.START_TOKEN]] + [self.char_indices[c] for c in cs] for cs in css
        ]

    def to_ids_padded(self, css, maxlen=None):
        return tf.keras.preprocessing.sequence.pad_sequences(self.to_ids(css), maxlen=CharacterTable.maxlen, dtype='int32',
                                                             padding='post', truncating='post',
                                                             value=self.char_indices[self.PAD_TOKEN])

    def __repr__(self):
        return repr(self.chars)


def unison_shuffled_copies(*texts):
    n = len(texts[0])
    for t in texts:
        assert len(t) == n
    p = np.random.permutation(n)
    return [t[p] for t in texts]


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
    maxlen: int = None
    letters_size: int = None
    dagesh_size: int = None
    sin_size: int = None
    niqqud_size: int = None

    def merge(self, ts, ds=None, ss=None, ns=None):
        default = [['_'] * len(ts[0]) for x in range(len(ts))]
        texts = [[self.letters_table.indices_char[x] for x in line] for line in ts]
        dageshs = [[self.dagesh_table.indices_char[x] for x in xs] for xs in from_categorical(ds)] if ds is not None else default
        sins = [[self.sin_table.indices_char[x] for x in xs] for xs in from_categorical(ss)] if ss is not None else default
        niqquds = [[self.niqqud_table.indices_char[x] for x in xs] for xs in from_categorical(ns)] if ns is not None else default
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

                if d != '_':
                    sentence.append('\u05bc')
                if s != '_':
                    sentence.append(s)
                if n != '_':
                    sentence.append(n)
            res.append(''.join(sentence))
        return res


def load_file(batch_size, validation, filenames, maxlen=100) -> Data:
    data = Data()
    input_texts = []
    dagesh_texts = []
    sin_texts = []
    niqqud_texts = []
    for filename in filenames:
        with open(filename, encoding='utf-8') as f:
            part_input_texts, part_dagesh_texts, part_sin_texts, part_niqqud_texts = unzip_dotted_lines(f, maxlen=maxlen)
        input_texts.extend(part_input_texts)
        dagesh_texts.extend(part_dagesh_texts)
        sin_texts.extend(part_sin_texts)
        niqqud_texts.extend(part_niqqud_texts)
    CharacterTable.maxlen=maxlen
    data.letters_table = CharacterTable(''.join(x for xs in input_texts for x in xs))
    data.dagesh_table = CharacterTable(''.join(x for xs in dagesh_texts for x in xs))
    data.sin_table = CharacterTable(''.join(x for xs in sin_texts for x in xs))
    data.niqqud_table = CharacterTable(''.join(x for xs in niqqud_texts for x in xs))

    input_texts = data.letters_table.to_ids_padded(input_texts)
    dagesh_texts = data.dagesh_table.to_ids_padded(dagesh_texts)
    sin_texts = data.sin_table.to_ids_padded(sin_texts)
    niqqud_texts = data.niqqud_table.to_ids_padded(niqqud_texts)

    m = len(input_texts) // batch_size * batch_size
    input_texts = input_texts[:m]
    # input_texts = tf.keras.utils.to_categorical(input_texts)
    dagesh_texts = tf.keras.utils.to_categorical(dagesh_texts[:m])
    sin_texts = tf.keras.utils.to_categorical(sin_texts[:m])
    niqqud_texts = tf.keras.utils.to_categorical(niqqud_texts[:m])

    _, _, data.dagesh_size = dagesh_texts.shape
    _, _, data.sin_size = sin_texts.shape
    _, _, data.niqqud_size = niqqud_texts.shape
    _, data.maxlen, = input_texts.shape

    data.letters_size = len(data.letters_table)

    data.input_texts, data.dagesh_texts, data.sin_texts, data.niqqud_texts = unison_shuffled_copies(input_texts, dagesh_texts, sin_texts, niqqud_texts)

    v = int(m*(1-validation))
    data.input_texts, data.input_validation = data.input_texts[:v], data.input_texts[v:]
    data.dagesh_texts, data.dagesh_validation = data.dagesh_texts[:v], data.dagesh_texts[v:]
    data.sin_texts, data.sin_validation = data.sin_texts[:v], data.sin_texts[v:]
    data.niqqud_texts, data.niqqud_validation = data.niqqud_texts[:v], data.niqqud_texts[v:]
    return data
