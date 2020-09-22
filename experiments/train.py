import numpy as np

from tensorflow.keras import layers
import tensorflow as tf

import wandb

import dataset
import schedulers

assert tf.config.list_physical_devices('GPU')


MAXLEN = 80
BATCH_SIZE = 64
UNITS = 512


def masked_metric(v, y_true):
    mask = tf.math.not_equal(y_true, 0)
    return tf.reduce_sum(tf.boolean_mask(v, mask)) / tf.cast(tf.math.count_nonzero(mask), tf.float32)


def masked_accuracy(y_true, y_pred):
    return masked_metric(tf.keras.metrics.sparse_categorical_accuracy(y_true, y_pred), y_true)


def sparse_categorical_crossentropy(y_true, y_pred):
    return masked_metric(tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True), y_true)


def get_xy(d):
    if d is None:
        return None
    d.shuffle()
    x = d.normalized
    y = {'N': d.niqqud, 'D': d.dagesh, 'S': d.sin }
    return (x, y)


def load_data():
    corpus = {
        'mix': dataset.read_corpora([
            'hebrew_diacritized_private/poetry',
            'hebrew_diacritized_private/rabanit',
            'hebrew_diacritized_private/pre_modern']
        ),
        'modern': dataset.read_corpora([
            'hebrew_diacritized/modern']
        )
    }

    data = {}
    np.random.seed(2)
    data['mix'] = dataset.load_data(corpus['mix'], validation_rate=0, maxlen=MAXLEN)
    np.random.seed(2)
    data['modern'] = dataset.load_data(corpus['modern'], validation_rate=0, maxlen=MAXLEN)

    return data


LETTERS_SIZE = len(dataset.letters_table)
NIQQUD_SIZE = len(dataset.niqqud_table)
DAGESH_SIZE = len(dataset.dagesh_table)
SIN_SIZE = len(dataset.sin_table)


def build_model(units):
    np.random.seed(2)
    tf.random.set_seed(2)

    inp = tf.keras.Input(shape=(None,), batch_size=None)
    embed = layers.Embedding(LETTERS_SIZE, units, mask_zero=True)(inp)

    layer = layers.Bidirectional(layers.LSTM(units, return_sequences=True, dropout=0.1), merge_mode='sum')(embed)
    layer = layers.Bidirectional(layers.LSTM(units, return_sequences=True, dropout=0.1), merge_mode='sum')(layer)
    layer = layers.Dense(units)(layer)

    outputs = [
        layers.Dense(NIQQUD_SIZE, name='N')(layer),
        layers.Dense(DAGESH_SIZE, name='D')(layer),
        layers.Dense(SIN_SIZE, name='S')(layer),
    ]
    return tf.keras.Model(inputs=inp, outputs=outputs)


def train():

    data = load_data()

    model = build_model(units=UNITS)
    model.compile(loss=sparse_categorical_crossentropy,
                  optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  metrics=masked_accuracy)

    model.save_weights('./checkpoints/uninit')

    config = {
        'batch_size': BATCH_SIZE,
        'maxlen': MAXLEN,
        'units': UNITS,
        'model': model,
        'order': [
            ('mix', (3e-3, 8e-3, 0), 'mix'),
            ('modern', (4e-3, 2e-3, 5e-4), 'modern'),
            ('modern', (27e-4, 11e-4, 8e-4), 'modern'),
            ('modern', (15e-4, 10e-4, 5e-4), 'modern'),
            ('modern', (12e-4, 8e-4, 5e-4), 'modern'),
            ('modern', (8e-4, 8e-4, 1e-4), 'final'),
        ],
    }

    run = wandb.init(project="dotter",
                     group="training",
                     name=f'train',
                     tags=[],
                     config=config)
    with run:
        for kind, clr, save in config['order']:
            train, validation = data[kind]

            training_data = (x, y) = get_xy(train)
            validation_data = get_xy(validation)

            wandb_callback = wandb.keras.WandbCallback(log_batch_frequency=10,
                                                       training_data=training_data,
                                                       validation_data=validation_data,
                                                       log_weights=True)

            scheduler = schedulers.CircularLearningRate(*clr)
            scheduler.set_dataset(train, BATCH_SIZE)

            model.fit(x, y, validation_data=validation_data,
                      batch_size=BATCH_SIZE, verbose=1,
                      callbacks=[wandb_callback, scheduler])

    model.save('./final_model/' + save + '.h5')


if __name__ == '__main__':
    train()
