import numpy as np

from tensorflow.keras import layers
import tensorflow as tf

import wandb

import dataset
import schedulers

assert tf.config.list_physical_devices('GPU')


class NakdimonParams:
    @property
    def name(self):
        return type(self).__name__

    maxlen = 80
    batch_size = 64
    units = 400

    corpus = {
        'mix': [
            'hebrew_diacritized/poetry',
            'hebrew_diacritized/rabanit',
            'hebrew_diacritized/pre_modern'
        ],
        'modern': [
            'hebrew_diacritized/modern',
            'hebrew_diacritized/dictaTestCorpus'
        ]
    }

    validation_rate = 0

    def epoch_params(self, data):
        yield ('mix', 1, schedulers.CircularLearningRate(3e-3, 8e-3, 0e-4, data['mix'][0], self.batch_size))

        lrs = [30e-4, 30e-4, 30e-4,  8e-4, 1e-4]
        yield ('modern', len(lrs), tf.keras.callbacks.LearningRateScheduler(lambda epoch, lr: lrs[epoch - 1]))

    def deterministic(self):
        np.random.seed(2)
        tf.random.set_seed(2)

    def build_model(self):
        # self.deterministic()

        inp = tf.keras.Input(shape=(None,), batch_size=None)
        embed = layers.Embedding(LETTERS_SIZE, self.units, mask_zero=True)(inp)

        layer = layers.Bidirectional(layers.LSTM(self.units, return_sequences=True, dropout=0.1), merge_mode='sum')(embed)
        layer = layers.Bidirectional(layers.LSTM(self.units, return_sequences=True, dropout=0.1), merge_mode='sum')(layer)
        layer = layers.Dense(self.units)(layer)

        outputs = [
            layers.Dense(NIQQUD_SIZE, name='N')(layer),
            layers.Dense(DAGESH_SIZE, name='D')(layer),
            layers.Dense(SIN_SIZE, name='S')(layer),
        ]
        return tf.keras.Model(inputs=inp, outputs=outputs)


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


def load_data(params: NakdimonParams):
    data = {}
    for stage_name, stage_dataset_filenames in params.corpus.items():
        np.random.seed(2)
        data[stage_name] = dataset.load_data(dataset.read_corpora(stage_dataset_filenames),
                                             validation_rate=params.validation_rate, maxlen=params.maxlen)
    return data


LETTERS_SIZE = len(dataset.letters_table)
NIQQUD_SIZE = len(dataset.niqqud_table)
DAGESH_SIZE = len(dataset.dagesh_table)
SIN_SIZE = len(dataset.sin_table)


def train(params: NakdimonParams):

    data = load_data(params)

    model = params.build_model()
    model.compile(loss=sparse_categorical_crossentropy,
                  optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  metrics=accuracy)

    config = {
        'batch_size': params.batch_size,
        'maxlen': params.maxlen,
        'units': params.units,
        'model': model,
    }

    run = wandb.init(project="dotter",
                     group="ablations",
                     name=params.name,
                     tags=[],
                     config=config)
    with run:
        last_epoch = 0
        for (stage, n_epochs, scheduler) in params.epoch_params(data):
            (train, validation) = data[stage]
            if validation:
                with open(f'validation_files_{stage}.txt', 'w') as f:
                    for p in validation.filenames:
                        print(p, file=f)

            training_data = (x, y) = get_xy(train)
            validation_data = get_xy(validation)

            wandb_callback = wandb.keras.WandbCallback(log_batch_frequency=10,
                                                       training_data=training_data,
                                                       validation_data=validation_data,
                                                       save_model=False,
                                                       log_weights=False)
            last_epoch += n_epochs
            model.fit(x, y, validation_data=validation_data,
                      epochs=last_epoch,
                      batch_size=params.batch_size, verbose=2,
                      callbacks=[wandb_callback, scheduler])
    return model


class Full(NakdimonParams):
    validation_rate = 0


if __name__ == '__main__':
    model = train(Full())
    model.save(f'./final_model/final.h5')
