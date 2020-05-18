# from keras examples
# https://www.tensorflow.org/tutorials/text/transformer
import tensorflow as tf

import time
import numpy as np


def get_angles(pos, i, d_model):
    return pos / np.power(10000, (2 * (i//2)) / np.float32(d_model))


def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)
    
    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    
    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        
    pos_encoding = angle_rads[np.newaxis, ...]
        
    return tf.cast(pos_encoding, dtype=tf.float32)


def create_look_ahead_mask(size):
    """create a diagonal matrix"""
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    # mask = tf.maximum(tf.eye(size), mask)
    return mask  # (seq_len, seq_len)
    # return tf.linalg.band_part(tf.ones((size, size)), tf.cast(0, size.dtype), size)
    # tf.constant(np.triu(np.ones(size)), dtype=tf.float32)
    # return 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)  # (seq_len, seq_len)


def scaled_dot_product_attention(q, k, v, mask=None, causal=False):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead) 
    but it must be broadcastable for addition.
    
    Args:
        q: query shape == (..., seq_len_q, depth)
        k: key shape == (..., seq_len_k, depth)
        v: value shape == (..., seq_len_v, depth_v)
        mask: Float tensor with shape broadcastable 
            to (..., seq_len_q, seq_len_k). Defaults to None.
        
    Returns:
        output, attention_weights
    """
    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    if causal:
        size = tf.shape(k)[-2]
        look_ahead_mask = create_look_ahead_mask(size)
        if mask is None:
            mask = look_ahead_mask
        else:
            mask = tf.maximum(mask, look_ahead_mask)

    if mask is not None:
        scaled_attention_logits += (mask * -1e17)
    
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    # if causal:
    #     # This fixes the first row, since softmax gives all the (very small) values similar probability
    #     attention_weights *= 1 - look_ahead_mask
    
    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, causal=False):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        
        assert d_model % self.num_heads == 0
        
        self.depth = d_model // self.num_heads
        
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
        
        self.dense = tf.keras.layers.Dense(d_model)

        self.causal = causal
            
    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
        
    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]
        
        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)
        
        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)
        
        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask, self.causal)
        
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)
            
        return output, attention_weights


def pointwise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(d_model)                  # (batch_size, seq_len, d_model)
    ])


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super().__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = pointwise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        
    def call(self, x, training, mask):

        attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)
        
        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)
        
        return out2


class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                maximum_position_encoding, rate=0.1):
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        
        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, self.d_model)
        
        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]
    
        self.dropout = tf.keras.layers.Dropout(rate)
            
    def call(self, x, training, mask):

        seq_len = tf.shape(x)[1]
        
        # adding embedding and position encoding.
        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)
        
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)
        
        return x  # (batch_size, input_seq_len, d_model)


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super().__init__()

        self.mha1 = MultiHeadAttention(d_model, num_heads, causal=True)
        self.mha2 = MultiHeadAttention(d_model, num_heads, causal=False)

        self.ffn = pointwise_feed_forward_network(d_model, dff)
    
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)
        
        
    def call(self, y, enc_output, training, dec_target_padding_mask, padding_mask):
        # enc_output.shape == (batch_size, input_seq_len, d_model)

        attn1, attn_weights_block1 = self.mha1(y, y, y, dec_target_padding_mask)  # (batch_size, target_seq_len, d_model)
        
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + y)  #  ! attn1 + y is a leak?

        attn2, attn_weights_block2 = self.mha2(enc_output, enc_output, out1, padding_mask)  # (batch_size, target_seq_len, d_model)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)

        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)

        return out3, attn_weights_block1, attn_weights_block2


class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size,
                 maximum_position_encoding, rate=0.1):
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        
        self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)
        
        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)
        
    def call(self, y, enc_output, training, dec_target_padding_mask, padding_mask):

        seq_len = tf.shape(y)[1]
        attention_weights = {}
        
        y = self.embedding(y)  # (batch_size, target_seq_len, d_model)
        y *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        y += self.pos_encoding[:, :seq_len, :]
        
        y = self.dropout(y, training=training)

        for i in range(self.num_layers):
            y, block1, block2 = self.dec_layers[i](y, enc_output, training, dec_target_padding_mask, padding_mask)
            
            attention_weights['decoder_layer{}_block1'.format(i+1)] = block1
            attention_weights['decoder_layer{}_block2'.format(i+1)] = block2
        
        # x.shape == (batch_size, target_seq_len, d_model)
        return y, attention_weights


loss_tracker = tf.keras.metrics.Mean(name="loss")


class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, 
                target_vocab_size, maximum_position_encoding_input, maximum_position_encoding_target, rate=0.1):
        super().__init__()

        self.encoder = Encoder(num_layers, d_model, num_heads, dff, 
                               input_vocab_size, maximum_position_encoding_input, rate)

        self.decoder = Decoder(num_layers, d_model, num_heads, dff, 
                              target_vocab_size, maximum_position_encoding_target, rate)

        self.dense = tf.keras.layers.Dense(target_vocab_size)
        self.softmax = tf.keras.layers.Softmax()
        self.masked_loss = MaskedCategoricalCrossentropy()

    def pseudo_build(self, input_maxlen, output_maxlen):
        # pseudo "build" step, to allow printing a summary:
        x = np.ones((1, input_maxlen), dtype=int)
        y = np.ones((1, output_maxlen+1), dtype=int)
        return self.train_step(x, y)

    def call(self, x, y, training, dec_target_padding_mask, padding_mask):
        enc_output = self.encoder(x, training, padding_mask)  # (batch_size, x_seq_len, d_model)
        
        # dec_output.shape == (batch_size, y_seq_len, d_model)
        dec_output, attention_weights = self.decoder(y, enc_output, training, dec_target_padding_mask, padding_mask)
        
        dense = self.dense(dec_output)  # (batch_size, y_seq_len, target_vocab_size)
        final_output = self.softmax(dense)
        return final_output, attention_weights

    train_step_signature = [
        tf.TensorSpec(shape=(None, None), dtype=tf.int64),
        tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    ]

    @tf.function(input_signature=train_step_signature)
    def train_step(self, x, y):
        y_inp = y[:, :-1]
        y = y[:, 1:]

        padding_mask = create_padding_mask(x)
        dec_target_padding_mask = create_padding_mask(y_inp)

        with tf.GradientTape() as tape:
            y_pred, _ = self(x, y_inp, True, dec_target_padding_mask, padding_mask)
            loss = self.compiled_loss(y, y_pred)

        gradients = tape.gradient(loss, self.trainable_variables)    
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.compiled_metrics.update_state(y, y_pred)

        return {m.name: m.result() for m in self.metrics}


    def test_step(self, x, y, sample_weight=None):
        y_pred = np.ones(x.shape, dtype=int)
        y = y[:, 1:]

        padding_mask = create_padding_mask(x)
        dec_target_padding_mask = create_padding_mask(x)
        for i in range(y_pred.shape[-1]):
            predictions, _ = self(x, y_pred, False, dec_target_padding_mask, padding_mask)
            predicted_id = tf.argmax(predictions, axis=-1)
            y_pred[:, i] = predicted_id.numpy()[:, i]

        # Updates stateful loss metrics.
        self.compiled_loss(y, y_pred, sample_weight, regularization_losses=self.losses)

        self.compiled_metrics.update_state(y, y_pred, sample_weight)
        return {m.name: m.result() for m in self.metrics}


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super().__init__()
        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = warmup_steps
        
    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


class MaskedCategoricalCrossentropy(tf.keras.losses.Loss):
    def __call__(self, y_true, y_pred, sample_weight=None):
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)

        mask = tf.cast(tf.math.logical_not(tf.math.equal(y_true, 0)), dtype=loss.dtype)
        loss *= mask

        return tf.reduce_sum(loss) / tf.reduce_sum(mask)


def masked_accuracy(real, pred):
    acc = tf.keras.metrics.sparse_categorical_accuracy(real, pred)

    mask = tf.cast(tf.math.logical_not(tf.math.equal(real, 0)), dtype=acc.dtype)
    acc *= mask

    return tf.reduce_sum(acc) / tf.reduce_sum(mask)


def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    # add extra dimensions to add the padding to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)
