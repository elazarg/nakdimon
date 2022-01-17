import numpy as np

from tensorflow.keras import layers
import tensorflow as tf

import wandb

import dataset
from dataset import NIQQUD_SIZE, DAGESH_SIZE, SIN_SIZE, LETTERS_SIZE
import schedulers

import transformer
from pathlib import Path
assert tf.config.list_physical_devices('GPU')

VALIDATION_PATH = 'hebrew_diacritized/validation/modern'
MAXLEN = 80


def masked_metric(v, y_true):
    mask = tf.math.not_equal(y_true, 0)
    return tf.reduce_sum(tf.boolean_mask(v, mask)) / tf.cast(tf.math.count_nonzero(mask), tf.float32)


def accuracy(y_true, y_pred):
    return masked_metric(tf.keras.metrics.sparse_categorical_accuracy(y_true, y_pred), y_true)


class NakdimonParams:
    @property
    def name(self):
        return f'{type(self).__name__}({self.units})'

    batch_size = 128
    units = 400
    corpus = {
        'mix': tuple([
            'hebrew_diacritized/poetry',
            'hebrew_diacritized/rabanit',
            'hebrew_diacritized/pre_modern',
            'hebrew_diacritized/shortstoryproject_predotted'
        ]),
        'dicta': tuple([
            'hebrew_diacritized/shortstoryproject_Dicta',
        ]),
        'modern': tuple([
            'hebrew_diacritized/modern',
            'hebrew_diacritized/dictaTestCorpus',
            'hebrew_diacritized/new',
            'hebrew_diacritized/validation'
        ])
    }

    validation_rate = 0

    subtraining_rate = {'mix': 1, 'modern': 1}

    def loss(self, y_true, y_pred):
        return masked_metric(tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True), y_true)

    def epoch_params(self, data):
        yield ('mix', 1, schedulers.CircularLearningRate(3e-3, 8e-3, 1e-4, data['mix'][0], self.batch_size))
        lrs1 = [30e-4, 10e-4]
        yield ('dicta', len(lrs1), tf.keras.callbacks.LearningRateScheduler(lambda epoch, lr: lrs1[epoch-1]))
        lrs2 = [10e-4, 10e-4, 3e-4]
        yield ('modern', len(lrs2), tf.keras.callbacks.LearningRateScheduler(lambda epoch, lr: lrs2[epoch-len(lrs1)-1]))

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
        return model


class TrainingParams(NakdimonParams):
    corpus = {
        'mix': tuple([
            'hebrew_diacritized/poetry',
            'hebrew_diacritized/rabanit',
            'hebrew_diacritized/pre_modern',
            'hebrew_diacritized/shortstoryproject_predotted'
        ]),
        'dicta': tuple([
            'hebrew_diacritized/shortstoryproject_Dicta',
        ]),
        'modern': tuple([
            'hebrew_diacritized/modern',
            'hebrew_diacritized/dictaTestCorpus',
            'hebrew_diacritized/new'
        ])
    }

    def __init__(self, units=NakdimonParams.units):
        self.units = units


class Transformer(TrainingParams):
    batch_size = 64

    def build_model(self) -> tf.keras.Model:
        from tensorflow.keras import layers

        inp = tf.keras.Input(shape=(None,), batch_size=None)

        layer = transformer.TokenAndPositionEmbedding(80, LETTERS_SIZE, self.units)(inp)
        layer = transformer.TransformerBlock(embed_dim=self.units, num_heads=4, ff_dim=self.units, rate=0.1)(layer)
        layer = layers.Dense(self.units, activation='relu')(layer)
        layer = transformer.TransformerBlock(embed_dim=self.units, num_heads=4, ff_dim=self.units, rate=0.1)(layer)
        layer = layers.Dense(self.units, activation='relu')(layer)
        outputs = [
            layers.Dense(NIQQUD_SIZE, name='N')(layer),
            layers.Dense(DAGESH_SIZE, name='D')(layer),
            layers.Dense(SIN_SIZE, name='S')(layer),
        ]
        return tf.keras.Model(inputs=inp, outputs=outputs)

    def epoch_params(self, data):
        lrs0 = [20e-4, 5e-4]
        yield ('mix', len(lrs0), tf.keras.callbacks.LearningRateScheduler(lambda epoch, lr: lrs0[epoch]))
        lrs1 = [10e-4, 10e-4]
        yield ('dicta', len(lrs1), tf.keras.callbacks.LearningRateScheduler(lambda epoch, lr: lrs1[epoch-len(lrs0)]))
        lrs2 = [10e-4] * 6
        yield ('modern', len(lrs2), tf.keras.callbacks.LearningRateScheduler(lambda epoch, lr: lrs2[epoch-len(lrs0)-len(lrs1)]))


class TwoLevelLSTM(TrainingParams):
    batch_size = 128

    def build_model(self) -> tf.keras.Model:
        from tensorflow.keras import layers

        inp = tf.keras.Input(shape=(None,), batch_size=None)

        layer = layers.Embedding(LETTERS_SIZE, self.units, mask_zero=True)(inp)
        layer = layers.Bidirectional(layers.LSTM(self.units, return_sequences=True, dropout=0.1), merge_mode='sum')(layer)
        layer = layers.Bidirectional(layers.LSTM(self.units, return_sequences=True, dropout=0.1), merge_mode='sum')(layer)
        outputs = [
            layers.Dense(NIQQUD_SIZE, name='N')(layer),
            layers.Dense(DAGESH_SIZE, name='D')(layer),
            layers.Dense(SIN_SIZE, name='S')(layer),
        ]
        return tf.keras.Model(inputs=inp, outputs=outputs)

    def epoch_params(self, data):
        lrs0 = [3e-4]
        yield ('mix', 1, schedulers.CircularLearningRate(1e-4, 8e-3, 1e-4, data['mix'][0], self.batch_size))
        lrs1 = [30e-4, 10e-4]
        yield ('dicta', len(lrs1), tf.keras.callbacks.LearningRateScheduler(lambda epoch, lr: lrs1[epoch-len(lrs0)]))
        lrs2 = [10e-4, 10e-4, 3e-4]
        yield ('modern', len(lrs2), tf.keras.callbacks.LearningRateScheduler(lambda epoch, lr: lrs2[epoch-len(lrs1)-len(lrs0)]))


def get_xy(d):
    x = d.normalized
    y = {'N': d.niqqud, 'D': d.dagesh, 'S': d.sin}
    return (x, y)


def load_data(params: NakdimonParams):
    train_dict = {}
    for stage_name, stage_dataset_filenames in params.corpus.items():
        train_dict[stage_name] = get_xy(dataset.load_data(tuple(stage_dataset_filenames), maxlen=MAXLEN).shuffle())
    return train_dict


def load_validation_data():
    return get_xy(dataset.load_data(tuple([VALIDATION_PATH]), maxlen=MAXLEN).shuffle())


def ablation_metrics(model):
    import nakdimon, metrics, hebrew

    def calculate_metrics(model, validation_path):
        for path in Path(validation_path).glob('*'):
            print(path, ' ' * 30, end='\r', flush=True)
            doc = metrics.read_document('expected', path)
            yield metrics.all_metrics(metrics.DocumentPack(
                path.parent.name,
                path.name,
                {
                    'expected': doc,
                    'actual': metrics.Document('actual', 'Nakdimon', nakdimon.predict(model, hebrew.remove_niqqud(doc.text), maxlen=MAXLEN))
                }
            ))

    return metrics.metricwise_mean(calculate_metrics(model, VALIDATION_PATH))


def train(params: NakdimonParams, group, ablation=False):
    print("Loading data...")
    train_dict = load_data(params)
    validation_data = load_validation_data() if ablation else None
    print("Creating model...")
    model = params.build_model()
    model.compile(loss=params.loss,
                  optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  metrics=accuracy)

    config = {
        'batch_size': params.batch_size,
        'units': params.units,
        'model': model,
    }

    run = wandb.init(project="dotter",
                     group=group,
                     name=params.name,
                     tags=[],
                     config=config)

    model = params.initialize_weights(model)

    with run:
        last_epoch = 0
        for phase, (stage, n_epochs, scheduler) in enumerate(params.epoch_params(train_dict)):
            training_data = (x, y) = train_dict[stage]

            wandb_callback = wandb.keras.WandbCallback(log_batch_frequency=10,
                                                       training_data=training_data,
                                                       validation_data=validation_data,
                                                       save_model=False,
                                                       log_weights=False,
                                                       save_graph=False)

            model.fit(x, y, validation_data=validation_data,
                      initial_epoch=last_epoch,
                      epochs=last_epoch + n_epochs,
                      batch_size=params.batch_size, verbose=2,
                      callbacks=[wandb_callback, scheduler])
            last_epoch += n_epochs
            if ablation:
                wandb.log({'epoch': last_epoch, **ablation_metrics(model)})
        if ablation:
            wandb.log({'final': 1, **ablation_metrics(model)})
    return model


def train_ablation(params, group):
    model = train(params, group, ablation=True)
    model.save(f'./models/ablations/{params.name}.h5')


class Full(NakdimonParams):
    validation_rate = 0


if __name__ == '__main__':
    model = train(Full(), 'Full', ablation=False)
    model.save(f'./models/Full.h5')
