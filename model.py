from json import decoder, encoder
import tensorflow as tf
from tensorflow.keras.layers import Dense, LayerNormalization, Dropout, Activation, GlobalAveragePooling1D

class EmbeddingsLayer(tf.keras.layers.Layer):
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