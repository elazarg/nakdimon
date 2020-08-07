import tensorflow as tf
import re
from tensorflow.keras.models import load_model

import utils
import dataset
import hebrew

# tf.config.set_visible_devices([], 'GPU')

model = load_model('models/modern.h5')


def naqdan(data: dataset.Data) -> str:
    prediction = model.predict(data.normalized)
    [actual_niqqud, actual_dagesh, actual_sin] = [dataset.from_categorical(prediction[0]), dataset.from_categorical(prediction[1]), dataset.from_categorical(prediction[2])]
    actual = dataset.merge(data.text, data.normalized, actual_niqqud, actual_dagesh, actual_sin)
    return ' '.join(actual)


def diacritize_text(text):
    return naqdan(dataset.Data.from_text(hebrew.iterate_dotted_text(text), 90))


def diacritize_file(input_filename='-', output_filename='-'):
    with utils.smart_open(input_filename, 'r', encoding='utf-8') as f:
        text = f.read()
        text = re.sub('[\u05b0-\u05bc]', '', text)
    text = diacritize_text(text)
    with utils.smart_open(output_filename, 'w', encoding='utf-8') as f:
        f.write(text)


if __name__ == '__main__':
    diacritize_file('../Neural-Sentiment-Analyzer-for-Modern-Hebrew/data/token_train.tsv',
                    '../Neural-Sentiment-Analyzer-for-Modern-Hebrew/data/token_train_dotted.tsv')
