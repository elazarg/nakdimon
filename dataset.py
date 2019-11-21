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


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def from_categorical(t):
    return np.argmax(t, axis=-1)


@dataclass
class Data:
    input_text: np.ndarray = None
    niqqud_texts: np.ndarray = None
    input_validation: np.ndarray = None
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


def load_file(batch_size, validation=0.2, filenames=['bible_text/bible.txt']) -> Data:
    data = Data()
    input_texts = []
    niqqud_texts = []
    for filename in filenames:
        with open(filename, encoding='utf-8') as f:
            part_input_texts, _, _, part_niqqud_texts = scrape_bible.unzip_dotted_lines(f)
        input_texts.extend(part_input_texts)
        niqqud_texts.extend(part_niqqud_texts)
    data.letters_table = CharacterTable(''.join(x for xs in input_texts for x in xs))
    data.niqqud_table = CharacterTable(''.join(x for xs in niqqud_texts for x in xs))

    input_texts = data.letters_table.to_ids_padded(input_texts)
    niqqud_texts = data.niqqud_table.to_ids_padded(niqqud_texts)

    m = len(input_texts) // batch_size * batch_size
    input_texts = input_texts[:m]
    # input_texts = tf.keras.utils.to_categorical(input_texts)
    niqqud_texts = tf.keras.utils.to_categorical(niqqud_texts[:m])

    _, _, data.niqqud_size = niqqud_texts.shape
    _, data.maxlen, = input_texts.shape

    data.letters_size = len(data.letters_table)

    data.input_texts, data.niqqud_texts = unison_shuffled_copies(input_texts, niqqud_texts)

    v = int(m*(1-validation))
    data.input_texts, data.input_validation = data.input_texts[:v], data.input_texts[v:]
    data.niqqud_texts, data.niqqud_validation = data.niqqud_texts[:v], data.niqqud_texts[v:]
    return data
