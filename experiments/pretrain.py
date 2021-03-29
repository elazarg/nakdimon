import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import wandb

from dataset import letters_table, NIQQUD_SIZE, DAGESH_SIZE, SIN_SIZE, LETTERS_SIZE
import utils
import hebrew
from train import TrainingParams
import metrics


pretrain_path = f'./models/wiki'
model_name = pretrain_path + 'pretrain.h5'


class BaseModel(keras.Model):
    def __init__(self, units, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.units = units
        self.embed = layers.Embedding(LETTERS_SIZE, units, mask_zero=True)

        self.lstm1 = layers.Bidirectional(layers.LSTM(units, return_sequences=True, dropout=0.1), merge_mode='sum')
        self.lstm2 = layers.Bidirectional(layers.LSTM(units, return_sequences=True, dropout=0.1), merge_mode='sum')

    def call(self, layer):
        layer = self.embed(layer)
        layer = self.lstm1(layer)
        layer = self.lstm2(layer)
        return layer

    def get_config(self):
        return {
            # 'units': self.units,
            'embed': self.embed,
            'lstm1': self.lstm1,
            'lstm2': self.lstm2,
        }

    @classmethod
    def from_config(cls, config, custom_objects=None):
        model = cls(400)  # config['units'])
        model.embed = config['embed']
        model.lstm1 = config['lstm1']
        model.lstm2 = config['lstm2']
        return model


def self_supervized_model(units):
    return keras.Sequential([
        BaseModel(units, name="base"),
        layers.Dense(LETTERS_SIZE)
    ])


class Pretrained(TrainingParams):
    def build_model(self):
        inp = keras.Input(shape=(None,), batch_size=None)

        pretrained = keras.models.load_model(model_name,
                                             custom_objects={'loss': self.loss, 'BaseModel': BaseModel(self.units)})
        base = pretrained.layers[0]

        embed = base.embed(inp)
        layer = base.lstm1(embed)
        layer = base.lstm2(layer)

        layer = pretrained.layers[1](layer)
        outputs = [
            layers.Dense(NIQQUD_SIZE, name='N')(layer),
            layers.Dense(DAGESH_SIZE, name='D')(layer),
            layers.Dense(SIN_SIZE, name='S')(layer),
        ]
        return keras.Model(inputs=inp, outputs=outputs)


class PretrainedModernOnly(Pretrained):
    def epoch_params(self, data):
        lrs = [30e-4, 30e-4, 30e-4, 8e-4, 1e-4]
        yield ('modern', len(lrs), keras.callbacks.LearningRateScheduler(lambda epoch, lr: lrs[epoch]))


def get_masked(raw_y, rate):
    mask = np.full(raw_y.shape[0] * raw_y.shape[1], 1)
    mask[:round(len(mask) * rate)] = 0
    np.random.shuffle(mask)
    mask.resize(raw_y.shape, refcheck=False)
    x = raw_y * mask
    y = raw_y  #  * (1 - mask)
    return x, y


def load_plaintext(filename, maxlen):
    import math
    with open(filename, encoding='utf8') as f:
        text = f.read()
    res = np.array([letters_table.char_indices[hebrew.normalize(c)] for c in text])
    res.resize((math.ceil(len(res) / maxlen), maxlen), refcheck=False)
    return res


def pretrain():
    model = self_supervized_model(400)
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=keras.optimizers.Adam(learning_rate=8e-5),
                  metrics='accuracy')

    config = {
        'batch_size': 128,
        'maxlen': 80,
        'units': 400,
        'model': model,
    }

    run = wandb.init(project="dotter",
                     group="pretrain",
                     tags=[],
                     config=config)

    wandb_callback = wandb.keras.WandbCallback(log_batch_frequency=50,
                                               save_model=False,
                                               log_weights=False)

    with run:
        for fname in utils.iterate_files(["../wikipedia/AA"]):
            name = fname.split('/')[-1]
            raw_y = load_plaintext(fname, 80)
            utils.shuffle_in_unison(raw_y)
            x, y = get_masked(raw_y, 0.3)
            model.fit(x, y, batch_size=128, validation_split=0.1, callbacks=[wandb_callback])
            model.save(f'{pretrain_path}/{name}.h5', save_format='tf')

    model.save(model_name, save_format='tf')
    return model


def train_ablation(params):
    from train import train
    model = train(params)
    model.save(f'./models/ablations/{params.name}.h5')


if __name__ == '__main__':
    mode = ''

    if mode == 'pretrain':
        pretrain()
    # elif mode == 'train_ablation':
    #     train_ablation(PretrainedModernOnly())
    else:
        import ablations
        tf.config.set_visible_devices([], 'GPU')
        model_name = 'PretrainedModernOnly'
        model = tf.keras.models.load_model(f'models/ablations/{model_name}.h5',
                                           custom_objects={'loss': TrainingParams().loss})
        print(model_name, *metrics.metricwise_mean(ablations.calculate_metrics(model)).values(), sep=', ')

