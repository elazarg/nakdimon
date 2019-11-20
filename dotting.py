import tensorflow as tf
import tensorflow.keras.layers as layers
import numpy as np
import translation
import itertools
import dataset_maker

import gpu_utils
gpu_utils.setup_gpus()

BUFFER_SIZE = 10000
BATCH_SIZE = 512  # 512


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def from_categorical(t):
    return np.argmax(t, axis=-1)


EPOCHS = 19  # Number of epochs to train for.
latent_dim = 256  # Latent dimensionality of the encoding space.
seq_length = 100

with open('bible_text/bible.txt', encoding='utf-8') as f:
    input_texts, letters_table, niqqud_texts, niqqud_table = dataset_maker.load_bible_text(f)

m = len(input_texts) // BATCH_SIZE * BATCH_SIZE
input_texts = tf.keras.utils.to_categorical(input_texts[:m])
niqqud_texts = tf.keras.utils.to_categorical(niqqud_texts[:m])

_, _, niqqud_size = niqqud_texts.shape
_, maxlen, letters_size = input_texts.shape
print(m, maxlen, letters_size, niqqud_size)

input_texts, niqqud_texts = unison_shuffled_copies(input_texts, niqqud_texts)

model = tf.keras.Sequential([
    # tf.keras.layers.Embedding(letters_size, 64, input_length=maxlen, batch_input_shape=BATCH_SIZE),
    tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(256, return_sequences=True, stateful=True, recurrent_initializer='glorot_uniform'),
        input_shape=(maxlen, letters_size),
        batch_size=BATCH_SIZE
    ),
    # tf.keras.layers.Bidirectional(
    #     tf.keras.layers.GRU(32, return_sequences=True, stateful=True, recurrent_initializer='glorot_uniform')
    # ),
    tf.keras.layers.GRU(niqqud_size, return_sequences=True, recurrent_initializer='glorot_uniform'),
    # tf.keras.layers.LSTM(niqqud_size, return_sequences=True),
    # tf.keras.layers.Softmax()
])

# best: mean_squared_logarithmic_error, stateless, accuracy: 0.9467

model.compile(loss='mean_squared_logarithmic_error',
              optimizer='adam',
              metrics=['accuracy'])
model.build()
model.summary()

model.fit(input_texts, niqqud_texts,
          batch_size=BATCH_SIZE,
          epochs=EPOCHS,
          callbacks=[
              tf.keras.callbacks.EarlyStopping(monitor='loss', patience=0, verbose=1)
          ]
)

model.add(tf.keras.layers.Softmax())
p = model.predict(input_texts[0:BATCH_SIZE])
# print(p)
# np.set_printoptions(threshold=np.inf)
# print('input', from_categorical(input_texts[0:BATCH_SIZE]))
# print('prediction', from_categorical(p))
print(dataset_maker.from_ids(letters_table, niqqud_table, from_categorical(input_texts[0:BATCH_SIZE]), from_categorical(p)))
