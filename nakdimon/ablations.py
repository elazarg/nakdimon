import keras
import tensorflow as tf

import train
from train import TrainingParams, train_ablation
import schedulers


class FullTraining(TrainingParams):
    pass


class SingleLayer(TrainingParams):
    def build_model(self):
        from train import LETTERS_SIZE, NIQQUD_SIZE, DAGESH_SIZE, SIN_SIZE
        layers = tf.keras.layers

        inp = tf.keras.Input(shape=(None,), batch_size=None)
        embed = layers.Embedding(LETTERS_SIZE, self.units, mask_zero=True)(inp)

        layer = layers.Bidirectional(layers.LSTM(self.units, return_sequences=True, dropout=0.1), merge_mode='sum')(embed)
        layer = layers.Dense(self.units)(layer)

        outputs = [
            layers.Dense(NIQQUD_SIZE, name='N')(layer),
            layers.Dense(DAGESH_SIZE, name='D')(layer),
            layers.Dense(SIN_SIZE, name='S')(layer),
        ]
        return tf.keras.Model(inputs=inp, outputs=outputs)


def SingleLayerLarge():
    return SingleLayer(557)


class SplitSin(TrainingParams):
    def build_model(self):
        from train import LETTERS_SIZE, NIQQUD_SIZE, DAGESH_SIZE, SIN_SIZE
        layers = tf.keras.layers

        inp = tf.keras.Input(shape=(None,), batch_size=None)
        embed = layers.Embedding(LETTERS_SIZE, self.units, mask_zero=True)(inp)

        layer = layers.Bidirectional(layers.LSTM(self.units, return_sequences=True, dropout=0.1), merge_mode='sum')(embed)
        layer = layers.Bidirectional(layers.LSTM(self.units, return_sequences=True, dropout=0.1), merge_mode='sum')(layer)
        layer = layers.Dense(self.units)(layer)

        sin_layer = layers.Bidirectional(layers.LSTM(self.units, return_sequences=True, dropout=0.1), merge_mode='sum')(embed)

        outputs = [
            layers.Dense(NIQQUD_SIZE, name='N')(layer),
            layers.Dense(DAGESH_SIZE, name='D')(layer),
            layers.Dense(SIN_SIZE, name='S')(sin_layer),
        ]
        return tf.keras.Model(inputs=inp, outputs=outputs)


class UnmaskedLoss(TrainingParams):
    def loss(self, y_true, y_pred):
        return tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)


class ConstantLR(TrainingParams):
    def __init__(self, lr_string):
        super().__init__()
        self.lr = float(lr_string)
        self.lr_string = lr_string

    @property
    def name(self):
        return f'ConstantLR({self.lr_string})'

    def epoch_params(self, data):
        scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch, lr: self.lr)
        yield ('premodern', 1, scheduler)
        yield ('modern', 5, tf.keras.callbacks.LearningRateScheduler(lambda epoch, lr: self.lr))


class ModernOnly(TrainingParams):
    corpus = {
        'modern': [
            'hebrew_diacritized/modern',
            'hebrew_diacritized/dictaTestCorpus'
        ]
    }

    def epoch_params(self, data):
        lrs = [30e-4, 30e-4, 30e-4, 8e-4, 1e-4]
        yield ('modern', len(lrs), tf.keras.callbacks.LearningRateScheduler(lambda epoch, lr: lrs[epoch]))


class Quick(TrainingParams):
    def epoch_params(self, data):
        yield ('modern', 1, tf.keras.callbacks.LearningRateScheduler(lambda epoch, lr: 3e-3))


class Chunk(TrainingParams):
    def __init__(self, maxlen):
        super().__init__()
        self.maxlen = maxlen

    @property
    def name(self):
        return f'Chunk({self.maxlen})'


class Batch(TrainingParams):
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size

    @property
    def name(self):
        return f'Batch({self.batch_size})'


class FullNoMix(TrainingParams):
    corpus = {
        'automatic': tuple([
            'hebrew_diacritized/shortstoryproject_predotted',
            'hebrew_diacritized/shortstoryproject_Dicta',
        ]),
        'modern': tuple([
            'hebrew_diacritized/modern',
            'hebrew_diacritized/dictaTestCorpus'
        ])
    }

    def epoch_params(self, data):
        lrs1 = [30e-4, 10e-4]
        yield ('automatic', len(lrs1), tf.keras.callbacks.LearningRateScheduler(lambda epoch, lr: lrs1[epoch-1]))
        lrs2 = [10e-4, 10e-4, 3e-4]
        yield ('modern', len(lrs2), tf.keras.callbacks.LearningRateScheduler(lambda epoch, lr: lrs2[epoch-len(lrs1)-1]))


class FullOrdered(TrainingParams):
    corpus = {
        'rabanit': tuple([
            'hebrew_diacritized/rabanit',
        ]),
        'pre_modern': tuple([
            'hebrew_diacritized/pre_modern',
            'hebrew_diacritized/shortstoryproject_predotted',
        ]),
        'automatic': tuple([
            'hebrew_diacritized/shortstoryproject_Dicta',
        ]),
        'modern': tuple([
            'hebrew_diacritized/modern',
            'hebrew_diacritized/dictaTestCorpus'
        ])
    }

    def epoch_params(self, data):
        yield ('rabanit', 1, tf.keras.callbacks.LearningRateScheduler(lambda epoch, lr: 30e-4))
        yield ('pre_modern', 1, tf.keras.callbacks.LearningRateScheduler(lambda epoch, lr: 30e-4))
        yield ('automatic', 1, tf.keras.callbacks.LearningRateScheduler(lambda epoch, lr: 30e-4))
        yield ('modern', 3, tf.keras.callbacks.LearningRateScheduler(lambda epoch, lr: 30e-4))


class NoPredotted(TrainingParams):
    corpus = {
        'automatic': tuple([
            'hebrew_diacritized/shortstoryproject_predotted',
            'hebrew_diacritized/shortstoryproject_Dicta',
        ]),
        'modern': tuple([
            'hebrew_diacritized/modern',
            'hebrew_diacritized/dictaTestCorpus'
        ])
    }

    def epoch_params(self, data):
        lrs1 = [30e-4, 10e-4]
        yield ('automatic', len(lrs1), tf.keras.callbacks.LearningRateScheduler(lambda epoch, lr: lrs1[epoch-1]))
        lrs2 = [10e-4, 10e-4, 3e-4]
        yield ('modern', len(lrs2), tf.keras.callbacks.LearningRateScheduler(lambda epoch, lr: lrs2[epoch-len(lrs1)-1]))


class FullUpdated(TrainingParams):
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
            'hebrew_diacritized/new'
        ])
    }

    def epoch_params(self, data):
        yield ('premodern', 1, schedulers.CircularLearningRate(3e-3, 8e-3, 1e-4, data['premodern'][0], self.batch_size))
        lrs1 = [30e-4, 10e-4]
        yield ('automatic', len(lrs1), tf.keras.callbacks.LearningRateScheduler(lambda epoch, lr: lrs1[epoch-1]))
        lrs2 = [10e-4, 10e-4, 3e-4]
        yield ('modern', len(lrs2), tf.keras.callbacks.LearningRateScheduler(lambda epoch, lr: lrs2[epoch-len(lrs1)-1]))


class TasteModernFirst(FullUpdated):
    def epoch_params(self, data):
        yield ('modern', 1,  tf.keras.callbacks.LearningRateScheduler(lambda epoch, lr: 3e-3))
        yield ('premodern', 1, schedulers.CircularLearningRate(3e-3, 8e-3, 1e-4, data['premodern'][0], self.batch_size))
        lrs1 = [30e-4, 10e-4]
        yield ('automatic', len(lrs1), tf.keras.callbacks.LearningRateScheduler(lambda epoch, lr: lrs1[epoch-1-1]))
        lrs2 = [10e-4, 10e-4, 3e-4]
        yield ('modern', len(lrs2), tf.keras.callbacks.LearningRateScheduler(lambda epoch, lr: lrs2[epoch-len(lrs1)-1-1]))


if __name__ == '__main__':
    # units = 400
    print(train.Full().build_model().count_params())
    # for cls in [train.TwoLevelLSTM]:
    #     for i in range(1):
    #         print(cls(units).build_model().count_params())
    #         train_ablation(cls(units), group=f"{cls.__name__}:2022")

