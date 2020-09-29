import numpy as np

from tensorflow.keras import layers
import tensorflow as tf

import wandb

import dataset
import schedulers

assert tf.config.list_physical_devices('GPU')


MAXLEN = 80
BATCH_SIZE = 64
UNITS = 400


def masked_metric(v, y_true):
    mask = tf.math.not_equal(y_true, 0)
    return tf.reduce_sum(tf.boolean_mask(v, mask)) / tf.cast(tf.math.count_nonzero(mask), tf.float32)


def accuracy(y_true, y_pred):
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
                  metrics=accuracy)

    clr_mix = (3e-3, 8e-3, 0e-4)
    lrs = [30e-4, 30e-4, 30e-4,  8e-4, 1e-4]

    config = {
        'batch_size': BATCH_SIZE,
        'maxlen': MAXLEN,
        'units': UNITS,
        'model': model,
        'order': [
              ('mix',    0, 1, schedulers.CircularLearningRate(*clr_mix, data['mix'][0], BATCH_SIZE)),
              ('modern', 1, (1 + len(lrs)), tf.keras.callbacks.LearningRateScheduler(lambda epoch, lr: lrs[epoch - 1])),
        ],
    }

    run = wandb.init(project="dotter",
                     group="training",
                     name=f'final',
                     tags=[],
                     config=config)
    with run:
        for kind, initial_epoch, epochs, scheduler in config['order']:
            train, validation = data[kind]

            training_data = (x, y) = get_xy(train)
            validation_data = get_xy(validation)

            wandb_callback = wandb.keras.WandbCallback(log_batch_frequency=10,
                                                       training_data=training_data,
                                                       validation_data=validation_data,
                                                       save_model=True,
                                                       log_weights=True)

            model.fit(x, y, validation_data=validation_data,
                      epochs=epochs,
                      batch_size=BATCH_SIZE, verbose=1,
                      callbacks=[wandb_callback, scheduler])

    model.save('./final_model/final.h5')


if __name__ == '__main__':
    train()
