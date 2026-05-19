from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import tensorflow as tf
import wandb
from tensorflow.keras import layers

from nakdimon import dataset, schedulers, transformer
from nakdimon.config import MODELS_DIR
from nakdimon.dataset import DAGESH_SIZE, LETTERS_SIZE, NIQQUD_SIZE, SIN_SIZE

# assert tf.config.list_physical_devices('GPU')

VALIDATION_PATH = 'hebrew_diacritized/validation/modern'
MAXLEN = 80


def _make_loss() -> tf.keras.losses.Loss:
    return tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, ignore_class=0)


def _make_accuracy() -> tf.keras.metrics.Metric:
    return tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")


class NakdimonParams:
    @property
    def name(self):
        return f'{type(self).__name__}({self.units})'

    batch_size = 128
    units = 400
    corpus = {
        'premodern': tuple([
            'hebrew_diacritized/poetry',
            'hebrew_diacritized/rabanit',
            'hebrew_diacritized/pre_modern',
            'hebrew_diacritized/shortstoryproject_predotted'
        ]),
        'automatic': tuple([
            'hebrew_diacritized/shortstoryproject_Dicta',
        ]),
        'modern': tuple([
            'hebrew_diacritized/modern',
            'hebrew_diacritized/dictaTestCorpus',
            'hebrew_diacritized/validation'
        ])
    }

    validation_rate = 0

    subtraining_rate = {'premodern': 1, 'modern': 1}

    def loss(self):
        """Returns the keras Loss instance used to compile the model.
        Override in a subclass to swap in a different loss for an ablation."""
        return _make_loss()

    def epoch_params(self, data):
        yield ('premodern', 1, schedulers.CircularLearningRate(3e-3, 8e-3, 1e-4, data['premodern'][0], self.batch_size))
        lrs1 = [30e-4, 10e-4]
        yield ('automatic', len(lrs1), tf.keras.callbacks.LearningRateScheduler(lambda epoch, lr: lrs1[epoch-1]))
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
        'premodern': tuple([
            'hebrew_diacritized/poetry',
            'hebrew_diacritized/rabanit',
            'hebrew_diacritized/pre_modern',
            'hebrew_diacritized/shortstoryproject_predotted'
        ]),
        'automatic': tuple([
            'hebrew_diacritized/shortstoryproject_Dicta',
        ]),
        'modern': tuple([
            'hebrew_diacritized/modern',
            'hebrew_diacritized/dictaTestCorpus',
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
        yield ('premodern', len(lrs0), tf.keras.callbacks.LearningRateScheduler(lambda epoch, lr: lrs0[epoch]))
        lrs1 = [10e-4, 10e-4]
        yield ('automatic', len(lrs1), tf.keras.callbacks.LearningRateScheduler(lambda epoch, lr: lrs1[epoch-len(lrs0)]))
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
        yield ('premodern', 1, schedulers.CircularLearningRate(1e-4, 8e-3, 1e-4, data['premodern'][0], self.batch_size))
        lrs1 = [30e-4, 10e-4]
        yield ('automatic', len(lrs1), tf.keras.callbacks.LearningRateScheduler(lambda epoch, lr: lrs1[epoch-len(lrs0)]))
        lrs2 = [10e-4, 10e-4, 3e-4]
        yield ('modern', len(lrs2), tf.keras.callbacks.LearningRateScheduler(lambda epoch, lr: lrs2[epoch-len(lrs1)-len(lrs0)]))


def get_xy(d: dataset.Data):
    x = d.normalized
    y = {'N': d.niqqud, 'D': d.dagesh, 'S': d.sin}
    return (x, y)


def load_data(params: NakdimonParams):
    logging.info("Loading training data...")
    train_dict = {}
    for stage_name, stage_dataset_filenames in params.corpus.items():
        logging.info(f"Loading training data: {stage_name}...")
        data = dataset.load_data(tuple(stage_dataset_filenames), maxlen=MAXLEN)
        data.shuffle()
        train_dict[stage_name] = get_xy(data)
        logging.info(f"{stage_name} loaded.")
    logging.info("Training data loaded.")
    return train_dict


def load_validation_data():
    logging.info("Loading validation data...")
    data = dataset.load_data(tuple([VALIDATION_PATH]), maxlen=MAXLEN)
    data.shuffle()
    result = get_xy(data)
    logging.info("Validation data loaded.")
    return result


def ablation_metrics(model):
    from nakdimon import hebrew, metrics, predict

    def calculate_metrics(model, validation_path):
        for path in Path(validation_path).glob('*'):
            logging.debug(path)
            doc = metrics.read_document('expected', path)
            yield metrics.all_metrics(metrics.DocumentPack(
                path.parent.name,
                path.name,
                {
                    'expected': doc,
                    'actual': metrics.Document('actual', 'Nakdimon', predict.predict(hebrew.remove_niqqud(doc.text), model, maxlen=MAXLEN))
                }
            ))

    return metrics.metricwise_mean(calculate_metrics(model, VALIDATION_PATH))


def train(params: NakdimonParams, group, ablation=False, wandb_enabled=False):
    train_dict = load_data(params)
    validation_data = load_validation_data() if ablation else None
    logging.info("Creating model...")
    model = params.build_model()
    model.compile(
        loss=params.loss(),
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        metrics={"N": _make_accuracy(), "D": _make_accuracy(), "S": _make_accuracy()},
    )

    config = {
        'batch_size': params.batch_size,
        'units': params.units,
        'model': model,
    }

    run = wandb.init(project="dotter",
                     group=group,
                     name=params.name,
                     tags=[],
                     config=config,
                     mode="enabled" if wandb_enabled else "disabled")

    model = params.initialize_weights(model)

    with run:
        last_epoch = 0
        for phase, (stage, n_epochs, scheduler) in enumerate(params.epoch_params(train_dict)):
            logging.info(f"Training phase {phase}: {stage}, {n_epochs} epochs.")

            x, y = train_dict[stage]

            from wandb.integration.keras import WandbMetricsLogger
            wandb_callback = WandbMetricsLogger(log_freq=10)

            model.fit(x, y, validation_data=validation_data,
                      initial_epoch=last_epoch,
                      epochs=last_epoch + n_epochs,
                      batch_size=params.batch_size,  # verbose=2,
                      callbacks=[wandb_callback, scheduler])
            last_epoch += n_epochs
            if ablation:
                wandb.log({'epoch': last_epoch, **ablation_metrics(model)})
        if ablation:
            wandb.log({'final': 1, **ablation_metrics(model)})
    return model


def train_ablation(params, group):
    model = train(params, group, ablation=True)
    model.save(f"./{MODELS_DIR}/ablations/{params.name}.keras")


class Full(NakdimonParams):
    validation_rate = 0


def main(*, model_path: str, wandb: bool, ablation_name: str | None):
    if ablation_name is not None:
        from nakdimon import ablations
        params = vars(ablations)[ablation_name]()
        model = train(params, ablation_name, ablation=False, wandb_enabled=wandb)
    else:
        model = train(Full(), 'Full', ablation=False, wandb_enabled=wandb)
    model.save(model_path)
