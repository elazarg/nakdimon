from pathlib import Path

from train import NakdimonParams, train
import tensorflow as tf
import metrics
import hebrew


class TrainingParams(NakdimonParams):
    validation_rate = 0.1
    maxlen = 72

    @property
    def name(self):
        return f'({self.maxlen})' + type(self).__name__


class FullTraining(TrainingParams):

    @property
    def name(self):
        return f'Full({self.units})'

    def __init__(self, units=NakdimonParams.units):
        self.units = units


class SingleLayerSmall(TrainingParams):
    def build_model(self):
        from train import tf, layers, LETTERS_SIZE, NIQQUD_SIZE, DAGESH_SIZE, SIN_SIZE

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


class SingleLayerLarge(SingleLayerSmall):
    units = 557


class SplitSin(TrainingParams):
    units = 400

    def build_model(self):
        from train import tf, layers, LETTERS_SIZE, NIQQUD_SIZE, DAGESH_SIZE, SIN_SIZE

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
    maxlen = 72
    def loss(self, y_true, y_pred):
        return tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)


class ConstantLR(TrainingParams):
    def __init__(self, lr_string):
        self.lr = float(lr_string)
        self.lr_string = lr_string

    @property
    def name(self):
        return f'ConstantLR({self.lr_string})'

    def epoch_params(self, data):
        scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch, lr: self.lr)
        yield ('mix', 1, scheduler)
        yield ('modern', 5, tf.keras.callbacks.LearningRateScheduler(lambda epoch, lr: self.lr))


class ModernOnly(TrainingParams):
    def epoch_params(self, data):
        lrs = [30e-4, 30e-4, 30e-4, 8e-4, 1e-4]
        yield ('modern', len(lrs), tf.keras.callbacks.LearningRateScheduler(lambda epoch, lr: lrs[epoch]))


class Chunk(TrainingParams):
    def __init__(self, maxlen):
        self.maxlen = maxlen

    @property
    def name(self):
        return f'Chunk({self.maxlen})'


class Batch(TrainingParams):
    def __init__(self, batch_size):
        self.batch_size = batch_size

    @property
    def name(self):
        return f'Batch({self.batch_size})'


def calculate_metrics(model):
    import nakdimon
    for file in Path('./validation/expected/modern/').glob('*'):
        print(file, ' ' * 30, end='\r', flush=True)
        with open(file, encoding='utf8') as f:
            expected = metrics.cleanup(f.read())
        actual = metrics.cleanup(nakdimon.predict(model, hebrew.remove_niqqud(expected)))
        yield metrics.all_metrics(actual, expected)


def train_ablation(params):
    model = train(params)
    model.save(f'./models/ablations/72/{params.name}.h5')


if __name__ == '__main__':
    # train_ablation(ModernOnly())

    # train_ablation(ConstantLR('3e-4'))
    # train_ablation(ConstantLR('1e-3'))
    # train_ablation(ConstantLR('2e-3'))

    # train_ablation(SingleLayerSmall())
    # train_ablation(SingleLayerLarge())

    # train_ablation(UnmaskedLoss())
    # train_ablation(FullTraining(800))
    # train_ablation(ConstantLR('3e-3'))
    # train_ablation(Batch(128))
    #
    # train_ablation(Chunk(64))
    # train_ablation(SplitSin())

    import os
    tf.config.set_visible_devices([], 'GPU')
    for model_name in ['Full(800).h5',  # '(72)SingleLayerLarge.h5',
                       # '(72)SingleLayerSmall.h5', 'ConstantLR(1e-3).h5'
                       ]:
        model = tf.keras.models.load_model('models/ablations/72/' + model_name, custom_objects={'loss': NakdimonParams().loss})
        print(model_name, *metrics.metricwise_mean(calculate_metrics(model)).values(), sep=', ')
