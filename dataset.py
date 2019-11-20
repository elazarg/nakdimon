import tensorflow as tf
import translation
import numpy as np


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
        return tf.keras.preprocessing.sequence.pad_sequences(self.to_ids(css), maxlen=maxlen, dtype='int32',
                                                             padding='post', truncating='post',
                                                             value=self.char_indices[self.PAD_TOKEN])


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def from_categorical(t):
    return np.argmax(t, axis=-1)


class Data:
    input_texts = None
    niqqud_texts = None
    letters_table = None
    niqqud_table = None
    maxlen = None
    letters_size = None
    niqqud_size = None

    def merge(self, q, p):

        # for cs in css:
        #     del cs[cs.index(self.char_indices[self.PAD_TOKEN]):]
        #     del cs[0]
        texts = from_categorical(q)
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


def load_bible(batch_size) -> Data:
    data = Data()
    with open('bible_text/bible.txt', encoding='utf-8') as f:
        input_texts, _, _, niqqud_texts = translation.unzip_dotted_lines(f)
        data.letters_table = CharacterTable(''.join(x for xs in input_texts for x in xs))
        data.niqqud_table = CharacterTable(''.join(x for xs in niqqud_texts for x in xs))

        input_texts = data.letters_table.to_ids_padded(input_texts)
        niqqud_texts = data.niqqud_table.to_ids_padded(niqqud_texts)

    m = len(input_texts) // batch_size * batch_size
    input_texts = tf.keras.utils.to_categorical(input_texts[:m])
    niqqud_texts = tf.keras.utils.to_categorical(niqqud_texts[:m])

    _, _, data.niqqud_size = niqqud_texts.shape
    _, data.maxlen, data.letters_size = input_texts.shape

    data.input_texts, data.niqqud_texts = unison_shuffled_copies(input_texts, niqqud_texts)
    return data
