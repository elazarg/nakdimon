import tensorflow as tf
import re
from tensorflow.keras.models import load_model

import utils
import dataset
import hebrew

tf.config.set_visible_devices([], 'GPU')

model = load_model('models/modern.h5')


def merge_unconditional(texts, tnss, nss, dss, sss):
    res = []
    for ts, tns, ns, ds, ss in zip(texts, tnss, nss, dss, sss):
        sentence = []
        for t, tn, n, d, s in zip(ts, tns, ns, ds, ss):
            if tn == 0:
                break
            sentence.append(t)
            sentence.append(dataset.dagesh_table.indices_char[d] if hebrew.can_dagesh(t) else '\uFEFF')
            sentence.append(dataset.sin_table.indices_char[s] if hebrew.can_sin(t) else '\uFEFF')
            sentence.append(dataset.niqqud_table.indices_char[n] if hebrew.can_niqqud(t) else '\uFEFF')
        res.append(''.join(sentence))
    return res


def nakdimon(data: dataset.Data) -> str:
    prediction = model.predict(data.normalized)
    [actual_niqqud, actual_dagesh, actual_sin] = [dataset.from_categorical(prediction[0]), dataset.from_categorical(prediction[1]), dataset.from_categorical(prediction[2])]
    actual = merge_unconditional(data.text, data.normalized, actual_niqqud, actual_dagesh, actual_sin)
    return ' '.join(actual).replace('\ufeff', '').replace('  ', ' ')


def call_nakdimon(text: str) -> str:
    return nakdimon(dataset.Data.from_text(hebrew.iterate_dotted_text(text), 80)).replace(hebrew.RAFE, '')


def diacritize_file(input_filename='-', output_filename='-'):
    with utils.smart_open(input_filename, 'r', encoding='utf-8') as f:
        text = f.read()
        text = re.sub('[\u05b0-\u05bc]', '', text)
    text = call_nakdimon(text)
    with utils.smart_open(output_filename, 'w', encoding='utf-8') as f:
        f.write(text)


if __name__ == '__main__':
    diacritize_file('test/hillel.txt', '-')
