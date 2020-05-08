import tensorflow as tf

from tensorflow.keras.models import load_model

import utils
import dataset
import hebrew

tf.config.set_visible_devices([], 'GPU')

model = load_model('./modern.h5')


def naqdan(data: dataset.Data) -> str:
    prediction = model.predict(data.normalized)
    [actual_niqqud, actual_dagesh, actual_sin] = [dataset.from_categorical(prediction[0]), dataset.from_categorical(prediction[1]), dataset.from_categorical(prediction[2])]
    actual = dataset.merge(data.text, ts=data.normalized, ns=actual_niqqud, ds=actual_dagesh, ss=actual_sin)
    return '.\n'.join(' '.join(actual).split('. '))


def diacritize_text(text):
    return naqdan(dataset.Data.from_text(hebrew.iterate_dotted_text(text), 82))


def diacritize_file(input_filename='-', output_filename='-'):
    text = naqdan(dataset.load_file(input_filename, 82))
    with utils.smart_open(output_filename, 'w', encoding='utf-8') as f:
        f.write(text)


if __name__ == '__main__':
    diacritize_file('sample.txt')
