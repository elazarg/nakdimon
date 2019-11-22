import tensorflow as tf
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

#     1    4    4:         23.04 after 10 epochs, batch=64
#     2    8    8:         69.75 after 10 epochs, batch=64
#     4   16   16:         70.14 after 10 epochs, batch=64
#     8   32   32:         80.74 after 10 epochs, batch=64
#    16   64   64: 72.67, 76.20, 78.48, 79.45, 80.11, 86.75 after 10 epochs, batch=64
#    32  128  128:         88.2  after 10 epochs, batch=64
#    64  256  256:         91.2  after 10 epochs, batch=64
#   128  512  512:         92.0  after 10 epochs, batch=64
#   256 1024 1024:         92.96 after 10 epochs, batch=64
EMBED_DIM = 32
FACTOR = 4

inp = tf.keras.Input(shape=(data.maxlen,), batch_size=BATCH_SIZE)
h = tf.keras.layers.Embedding(data.letters_size, EMBED_DIM,  mask_zero=True)(inp)
h = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(EMBED_DIM*FACTOR, return_sequences=True), merge_mode='sum')(h)

h = tf.keras.layers.Dense(EMBED_DIM*FACTOR, activation='relu')(h)

# h_dagesh = tf.keras.layers.Dense(EMBED_DIM*FACTOR, activation='relu')(h)
# h_sin = tf.keras.layers.Dense(EMBED_DIM*FACTOR, activation='relu')(h)
# h = tf.keras.layers.Dense(EMBED_DIM*FACTOR, activation='relu')(h)

output_dagesh = tf.keras.layers.Dense(data.dagesh_size, name='Dagesh')(h)
output_sin = tf.keras.layers.Dense(data.sin_size, name='Sin')(h)
output_niqqud = tf.keras.layers.Dense(data.niqqud_size, name='Niqqud')(h)

model = tf.keras.Model(inputs=[inp], outputs=[output_dagesh, output_sin, output_niqqud])

model.compile(loss=['mean_squared_logarithmic_error', 'mean_squared_logarithmic_error', 'mean_squared_logarithmic_error'],
              lossWeights=[0.048, 0.002, 0.95],
              optimizer='adam',
              metrics=['accuracy'])
plot_model(model, to_file='model.png')
model.summary()

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
