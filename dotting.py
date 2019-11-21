import tensorflow as tf
from tensorflow.keras.utils import plot_model
import numpy as np
import dataset

import gpu_utils
tf.get_logger().setLevel('INFO')
gpu_utils.setup_gpus()

BUFFER_SIZE = 10000
BATCH_SIZE = 64  # 512
EPOCHS = 10


PRELOAD = False
TRAIN = True

data = dataset.load_file(BATCH_SIZE, 0.1, filenames=['bible_text/bible.txt', 'short_table/short_table.txt'])

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(data.letters_size, 32,  input_length=data.maxlen,
                              batch_size=BATCH_SIZE,
                              mask_zero=True),
    tf.keras.layers.Bidirectional(tf.keras.layers.GRU(128, return_sequences=True)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(data.niqqud_size),
])

# best: mean_squared_logarithmic_error, stateless

model.compile(loss='mean_squared_logarithmic_error',
              optimizer='adam',
              metrics=['accuracy'])
plot_model(model, to_file='model.png')
model.summary()

if PRELOAD:
    model.load_weights(tf.train.latest_checkpoint('niqqud_checkpoints/'))
if TRAIN:
    model.fit(data.input_texts, data.niqqud_texts,
              batch_size=BATCH_SIZE,
              epochs=EPOCHS,
              validation_data=(data.input_validation, data.niqqud_validation),
              callbacks=[
                  # tf.keras.callbacks.ModelCheckpoint(filepath='niqqud_checkpoints/ckpt_{epoch}', save_weights_only=True),
                  tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=3, verbose=1),
                  tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.2, patience=0, min_lr=0.001),
                  # tf.keras.callbacks.TensorBoard(log_dir='logs\\fit\\', histogram_freq=1),
                  # tf.keras.LambdaCallback(on_epoch_end=lambda batch, logs: print(batch))
              ]
              )

model.add(tf.keras.layers.Softmax())

q = data.input_texts[0:BATCH_SIZE]
p = model.predict(q)
results = data.merge(q, p)

for r in results:
    print(r)
