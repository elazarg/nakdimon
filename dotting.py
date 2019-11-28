import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.keras.utils import plot_model
import numpy as np
import dataset

import gpu_utils
tf.get_logger().setLevel('INFO')
gpu_utils.setup_gpus()

BUFFER_SIZE = 10000
BATCH_SIZE = 64  # 512
EPOCHS = 5


PRELOAD = False
TRAIN = True

data = dataset.load_file(BATCH_SIZE, 0.05,
                         filenames=['texts/bible.txt', 'texts/short_table.txt', 'texts/treasure_island.txt'])

EMBED_DIM = 128
F = 4
inp = tf.keras.Input(shape=(data.input_texts.shape[1],), batch_size=BATCH_SIZE)
h = layers.Embedding(len(data.letters_table), EMBED_DIM, mask_zero=True)(inp)
h = layers.Bidirectional(layers.GRU(EMBED_DIM*F, return_sequences=True), merge_mode='sum')(h)
# h = layers.Add()([h, h1])
for k in range(3):
    h = layers.Dropout(0.1)(h)
    h1 = layers.Dense(EMBED_DIM*F, activation='relu')(h)
    h = layers.Add()([h, h1])
    
h = layers.Dropout(0.1)(h)

# output_dagesh = layers.Dense(len(data.dagesh_table), name='Dagesh')(h)
# output_sin = layers.Dense(len(data.sin_table), name='Sin')(h)
h = tf.keras.layers.Dense(len(data.niqqud_table), name='Niqqud')(h)
model_niqqud = tf.keras.Model(inputs=[inp], outputs=[h])

adam = tf.keras.optimizers.Adam(learning_rate=0.003, beta_1=0.9, beta_2=0.999, amsgrad=False)

model_niqqud.compile(loss='mean_squared_logarithmic_error', optimizer=adam, metrics=['accuracy'])

plot_model(model_niqqud, to_file='model.png')
model_niqqud.summary()

if PRELOAD:
    model.load_weights(tf.train.latest_checkpoint('niqqud_checkpoints/'))
if TRAIN:
    model.fit(data.input_texts, [data.dagesh_texts, data.sin_texts, data.niqqud_texts],
              batch_size=BATCH_SIZE,
              epochs=EPOCHS,
              validation_data=(data.input_validation, [data.dagesh_validation, data.sin_validation, data.niqqud_validation]),
              callbacks=[
                  # tf.keras.callbacks.ModelCheckpoint(filepath='niqqud_checkpoints/ckpt_{epoch}', save_weights_only=True),
                  tf.keras.callbacks.EarlyStopping(monitor='Niqqud_accuracy', patience=3, verbose=1),
                  tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.2, patience=0, min_lr=0.001),
                  # tf.keras.callbacks.TensorBoard(log_dir='logs\\fit\\', histogram_freq=1),
                  # tf.keras.LambdaCallback(on_epoch_end=lambda batch, logs: print(batch))
              ]
              )

model = tf.keras.Model(inputs=[inp],
                       outputs=[tf.keras.layers.Softmax()(x) for x in (output_dagesh, output_sin, output_niqqud)])

q = data.input_texts[0:BATCH_SIZE]
[d, s, n] = model.predict(q)
results = data.merge(q, d, s, n)

for r in results:
    print(r)
