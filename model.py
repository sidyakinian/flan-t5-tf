from json import decoder, encoder
import tensorflow as tf
from tensorflow.keras.layers import Dense, LayerNormalization, Dropout, Activation, Layer

class EmbeddingsLayer(Layer):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embedding = tf.keras.layers.Embedding(vocab_size, d_model)
    
    def call(self, x: tf.Tensor) -> tf.Tensor:
        x = self.embedding(x)
        return x

class MLP(Layer):
    def __init__(self, d_ff: int, d_output: int, activation: Activation):
        super().__init__()
        self.fc1 = Dense(d_ff, activation=activation)
        self.fc2 = Dense(d_output)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        x = self.fc1(x)
        x = self.fc2(x)
        return x

class SelfAttention(Layer):
    def __init__(self, d_k: int):
        super().__init__()
        self.d_k = d_k

    def call(self, K: tf.Tensor, Q: tf.Tensor, V: tf.Tensor) -> tf.Tensor:
        # K, Q, V are all of shape (batch_size, n_heads, n, d_k)
        K = tf.transpose(K, perm=[0, 1, 3, 2]) # (batch_size, n_heads, d_k, n)
        QK = tf.matmul(Q, K) / tf.math.sqrt(tf.cast(self.d_k, dtype=tf.float32)) # shape (batch_size, n_heads, n, n)
        P_bar = tf.nn.softmax(QK, axis=-1)
        return tf.matmul(P_bar, V) # (batch_size, n_heads, n, d_k)