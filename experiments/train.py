import numpy as np

from tensorflow.keras import layers
import tensorflow as tf

import wandb

import dataset
from dataset import NIQQUD_SIZE, DAGESH_SIZE, SIN_SIZE, LETTERS_SIZE
import schedulers

assert tf.config.list_physical_devices('GPU')


def masked_metric(v, y_true):
    mask = tf.math.not_equal(y_true, 0)
    return tf.reduce_sum(tf.boolean_mask(v, mask)) / tf.cast(tf.math.count_nonzero(mask), tf.float32)


def accuracy(y_true, y_pred):
    return masked_metric(tf.keras.metrics.sparse_categorical_accuracy(y_true, y_pred), y_true)


class NakdimonParams:
    @property
    def name(self):
        return type(self).__name__

    batch_size = 64
    units = 400

    corpus = {
        'mix': (80, tuple([
            'hebrew_diacritized/poetry',
            'hebrew_diacritized/rabanit',
            'hebrew_diacritized/pre_modern'
        ])),
        'modern': (80, tuple([
            'hebrew_diacritized/modern',
            'hebrew_diacritized/dictaTestCorpus'
        ]))
    }

    validation_rate = 0

    subtraining_rate = {'mix': 1, 'modern': 1}

    def loss(self, y_true, y_pred):
        return masked_metric(tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True), y_true)

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

    def initialize_weights(self, model):
        return


class TrainingParams(NakdimonParams):
    validation_rate = 0.1


def get_xy(d):
    if d is None:
        return None
    d.shuffle()
    x = d.normalized
    y = {'N': d.niqqud, 'D': d.dagesh, 'S': d.sin }
    return (x, y)


def load_data(params: NakdimonParams):
    data = {}
    for stage_name, (maxlen, stage_dataset_filenames) in params.corpus.items():
        np.random.seed(2)
        data[stage_name] = dataset.load_data(tuple(dataset.read_corpora(tuple(stage_dataset_filenames))),
                                             validation_rate=params.validation_rate,
                                             maxlen=maxlen
                                             # ,subtraining_rate=params.subtraining_rate[stage_name]
                                             )
    return data


def train(params: NakdimonParams, group, ablation=None):

    data = load_data(params)

    model = params.build_model()
    model.compile(loss=params.loss,
                  optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  metrics=accuracy)

    config = {
        'batch_size': params.batch_size,
        'units': params.units,
        'model': model,
        # 'rate_modern': params.subtraining_rate['modern']
    }

    run = wandb.init(project="dotter",
                     group=group,
                     name=params.name,
                     tags=[],
                     config=config)

    params.initialize_weights(model)

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

            model.fit(x, y, validation_data=validation_data,
                      initial_epoch=last_epoch,
                      epochs=last_epoch + n_epochs,
                      batch_size=params.batch_size, verbose=2,
                      callbacks=[wandb_callback, scheduler])
            last_epoch += n_epochs
        if ablation is not None:
            wandb.log({'final': 0, **ablation(model)})
    return model


class Full(NakdimonParams):
    validation_rate = 0


class FullNew(NakdimonParams):
    validation_rate = 0.1

    corpus = {
        'mix': (80, tuple([
            'hebrew_diacritized/poetry',
            'hebrew_diacritized/rabanit',
            'hebrew_diacritized/pre_modern'
        ])),
        'modern': (80, tuple([
            'hebrew_diacritized/modern',
            'hebrew_diacritized/shortstoryproject',
            'hebrew_diacritized/dictaTestCorpus'
        ]))
    }


if __name__ == '__main__':
    model = train(FullNew(), 'FullNew')
    model.save(f'./models/FullNew.h5')
