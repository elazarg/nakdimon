import tensorflow as tf
import numpy as np
import dataset

import gpu_utils
tf.get_logger().setLevel('WARNING')
gpu_utils.setup_gpus()

BUFFER_SIZE = 10000
BATCH_SIZE = 512  # 512
EPOCHS = 1

PRELOAD = True
TRAIN = False

data = dataset.load_bible(BATCH_SIZE)

model = tf.keras.Sequential([
    tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(256, return_sequences=True, recurrent_initializer='glorot_uniform'),
        input_shape=(data.maxlen, data.letters_size),
        batch_size=BATCH_SIZE
    ),
    tf.keras.layers.GRU(data.niqqud_size, return_sequences=True, recurrent_initializer='glorot_uniform'),
])

# best: mean_squared_logarithmic_error, stateless

model.compile(loss='mean_squared_logarithmic_error',
              optimizer='adam',
              metrics=['accuracy'])
model.build()
model.summary()

if PRELOAD:
    model.load_weights(tf.train.latest_checkpoint('niqqud_checkpoints/'))
if TRAIN:
    model.fit(data.input_texts, data.niqqud_texts,
              batch_size=BATCH_SIZE,
              epochs=EPOCHS,
              callbacks=[
                  tf.keras.callbacks.ModelCheckpoint(filepath='niqqud_checkpoints/ckpt_{epoch}', save_weights_only=True),
                  tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3, verbose=1),
                  tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.2, patience=0, min_lr=0.001),
                  tf.keras.callbacks.TensorBoard(log_dir='niqqud_checkpoints\\', histogram_freq=1),
                  # tf.keras.LambdaCallback(on_epoch_end=lambda batch, logs: print(batch))
              ]
              )

model.add(tf.keras.layers.Softmax())

q = data.input_texts[0:BATCH_SIZE]
p = model.predict(q)
results = data.merge(data.input_texts[0:BATCH_SIZE], model.predict(q))

for r in results:
    print(r)
