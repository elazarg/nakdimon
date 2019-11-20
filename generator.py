import tensorflow as tf
import numpy as np
import os
import time
# TODO: split chat
import gpu_utils
gpu_utils.setup_gpus()
#
# path_to_file = tf.keras.utils.get_file('shakespeare.txt',
#                                        'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')

seq_length = 40

BATCH_SIZE = 64
BUFFER_SIZE = 10000
EPOCHS = 1
PRELOAD = True
TRAIN = False

with open('bible_text/bible.txt', encoding='utf-8') as f:
    text = f.read()
    # The unique characters in the file
    vocab = list(sorted(set(text)))

print('{} unique characters'.format(len(vocab)))

# Creating a mapping from unique characters to indices
char2idx = {u: i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

text_as_int = [char2idx[c] for c in text]

examples_per_epoch = len(text_as_int) / (seq_length+2)

char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
sequences = char_dataset.batch(seq_length+2, drop_remainder=True)


def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text


dataset = sequences.map(split_input_target)
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)


def build_model(batch_size):
    import tensorflow.keras.layers as layers
    vocab_size = len(vocab)
    inp = layers.Input(batch_input_shape=[batch_size, None])
    x = layers.Embedding(vocab_size, 32)(inp)
    x = layers.LSTM(1024, return_sequences=True, stateful=True, recurrent_initializer='glorot_uniform')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.GRU(1024, return_sequences=True, stateful=True, recurrent_initializer='glorot_uniform')(x)
    x = layers.Dense(vocab_size)(x)
    return tf.keras.Model(inputs=[inp], outputs=[x])


def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)


model = build_model(BATCH_SIZE)
model.summary()
model.compile(optimizer='adam', loss=loss)

# Directory where the checkpoints will be saved
checkpoint_dir = './langmodel_checkpoints'

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(checkpoint_dir, "ckpt_{epoch}"),
    save_weights_only=True
)

if PRELOAD:
    model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
if TRAIN:
    history = model.fit(dataset, epochs=EPOCHS,
                        callbacks=[
                            checkpoint_callback,
                            tf.keras.callbacks.EarlyStopping(monitor='loss', patience=0, verbose=1)
                        ])

tf.train.latest_checkpoint(checkpoint_dir)
model = build_model(batch_size=1)
model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
model.build(tf.TensorShape([1, None]))


def generate_text(model, start_string):
    # Evaluation step (generating text using the learned model)

    # Number of characters to generate
    num_generate = 1000

    # Converting our start string to numbers (vectorizing)
    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)

    # Empty string to store our results
    text_generated = []

    # Low temperatures results in more predictable text.
    # Higher temperatures results in more surprising text.
    # Experiment to find the best setting.
    temperature = 0.8

    # Here batch size == 1
    model.reset_states()
    for i in range(num_generate):
        predictions = model(input_eval)
        # remove the batch dimension
        predictions = tf.squeeze(predictions, 0)

        # using a categorical distribution to predict the word returned by the model
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()

        # We pass the predicted word as the next input to the model
        # along with the previous hidden state
        input_eval = tf.expand_dims([predicted_id], 0)

        text_generated.append(idx2char[predicted_id])

    return start_string + ''.join(text_generated)


print(generate_text(model, start_string="וַיְהִי "))
