import tensorflow as tf
import scrape_bible
import numpy as np
from dataclasses import dataclass


class CharacterTable:
    START_TOKEN = '!'
    PAD_TOKEN = '.'

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
        return tf.keras.preprocessing.sequence.pad_sequences(self.to_ids(css), maxlen=400, dtype='int32',
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


@dataclass
class Data:
    input_text: np.ndarray = None
    dagesh_texts: np.ndarray = None
    sin_texts: np.ndarray = None
    niqqud_texts: np.ndarray = None
    input_validation: np.ndarray = None
    dagesh_validation: np.ndarray = None
    sin_validation: np.ndarray = None
    niqqud_validation: np.ndarray = None
    letters_table: CharacterTable = None
    niqqud_table: CharacterTable = None
    maxlen: int = None
    letters_size: int = None
    niqqud_size: int = None

    def merge(self, texts, p):

        # for cs in css:
        #     del cs[cs.index(self.char_indices[self.PAD_TOKEN]):]
        #     del cs[0]
        # texts = from_categorical(q)
        niqquds = from_categorical(p)
        assert len(texts) == len(niqquds)
        res = []
        for i in range(len(texts)):
            sentence = []
            for ci, ni in zip(texts[i], niqquds[i]):
                if ci == self.letters_table.char_indices[self.letters_table.START_TOKEN]:
                    continue
                if ci == self.letters_table.char_indices[self.letters_table.PAD_TOKEN]:
                    break
                sentence.append(self.letters_table.indices_char[ci])
                n = self.niqqud_table.indices_char[ni]
                if n != '_':
                    sentence.append(n)
            res.append(''.join(sentence))
        return res


def load_file(batch_size, validation, filenames) -> Data:
    data = Data()
    input_texts = []
    dagesh_texts = []
    sin_texts = []
    niqqud_texts = []
    for filename in filenames:
        with open(filename, encoding='utf-8') as f:
            part_input_texts, part_dagesh_texts, part_sin_texts, part_niqqud_texts = scrape_bible.unzip_dotted_lines(f)
        input_texts.extend(part_input_texts)
        dagesh_texts.extend(part_dagesh_texts)
        sin_texts.extend(part_sin_texts)
        niqqud_texts.extend(part_niqqud_texts)
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
