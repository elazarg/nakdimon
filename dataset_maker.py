import tensorflow as tf
import translation


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


def from_ids(text_table, niqqud_table, texts, niqquds):
    # for cs in css:
    #     del cs[cs.index(self.char_indices[self.PAD_TOKEN]):]
    #     del cs[0]
    assert len(texts) == len(niqquds)
    res = []
    for i in range(len(texts)):
        sentence = []
        for ci, ni in zip(texts[i], niqquds[i]):
            if ci == text_table.char_indices[text_table.START_TOKEN]:
                continue
            if ci == text_table.char_indices[text_table.PAD_TOKEN]:
                break
            sentence.append(text_table.indices_char[ci])
            n = niqqud_table.indices_char[ni]
            if n != '_':
                sentence.append(n)
        res.append(''.join(sentence))
    return res


def sequence_to_tf_example(text_ids, niqqud_ids):
    ex = tf.train.SequenceExample()
    # Feature lists for the two sequential features of our example
    text_tokens = ex.feature_lists.feature_list["text_tokens"]
    for token in text_ids:
        text_tokens.feature.add().int64_list.value.append(token)

    niqqud_tokens = ex.feature_lists.feature_list["niqqud_tokens"]
    for token in niqqud_ids:
        niqqud_tokens.feature.add().int64_list.value.append(token)

    return ex


def load_bible_text(lines):
    input_texts, _, _, niqqud_texts = translation.unzip_dotted_lines(lines)
    letter_table = CharacterTable(''.join(x for xs in input_texts for x in xs))
    niqqud_table = CharacterTable(''.join(x for xs in niqqud_texts for x in xs))

    input_texts = letter_table.to_ids_padded(input_texts)
    niqqud_texts = niqqud_table.to_ids_padded(niqqud_texts)
    return input_texts, letter_table, niqqud_texts, niqqud_table


if __name__ == '__main__':
    with open('bible_text/bible.txt', encoding='utf-8') as f:
        input_texts, letter_table, niqqud_texts, niqqud_table = load_bible_text(f)
        print(input_texts[:25])
        print(niqqud_texts[:25])
        print(from_ids(letter_table, niqqud_table, input_texts[:25], niqqud_texts[:25]))
