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

def reshape_for_multihead_attention(M, batch_size, n, n_heads, d_k):
    M = tf.reshape(M, shape=[batch_size, n, n_heads, d_k])
    M = tf.transpose(M, perm=[0, 2, 1, 3]) # shape (batch_size, n_heads, n, d_k)
    return M

class MultiHeadSelfAttention(Layer):
    def __init__(self, d_k: int, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.d_k = d_k
        self.n_heads = n_heads
        self.self_attention = SelfAttention(d_k=self.d_k)
        self.w_k = Dense(self.d_k * n_heads) # equals d_model but more illustrative
        self.w_q = Dense(self.d_k * n_heads)
        self.w_v = Dense(self.d_k * n_heads)
        self.w_o = Dense(self.d_model)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        # shape of K, Q, V: (batch_size, n, d_model)
        assert tf.shape(K) == tf.shape(Q) == tf.shape(V), "K, Q, V must have the same shape" # Idk if tf.shape will work with XLA
        # TODO: find another way to determine shape other than passing batch size on model init
        batch_size, n, d_k = tf.shape(K).numpy()
        K, Q, V = self.w_k(x), self.w_q(x), self.w_v(x)

        K = reshape_for_multihead_attention(K, batch_size, n, self.n_heads, self.d_k)  
        Q = reshape_for_multihead_attention(Q, batch_size, n, self.n_heads, self.d_k)
        V = reshape_for_multihead_attention(V, batch_size, n, self.n_heads, self.d_k)
        
        attention_output = self.self_attention(K, Q, V)
        # Concat heads with transpose and reshape
        attention_output = tf.transpose(attention_output, perm=[0, 2, 1, 3]) # shape (batch_size, n, n_heads, d_k)
        attention_output = tf.reshape(attention_output, shape=[batch_size, n, self.d_model]) # shape (batch_size, n, d_model)
        outputs = self.w_o(attention_output)

        return outputs

class MultiHeadEncDecAttention(Layer):
    def __init__(self, d_k: int, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.d_k = d_k
        self.n_heads = n_heads
        self.self_attention = SelfAttention(d_k=self.d_k)
        self.w_k = Dense(self.d_k * n_heads) # equals d_model but more illustrative
        self.w_q = Dense(self.d_k * n_heads)
        self.w_v = Dense(self.d_k * n_heads)
        self.w_o = Dense(self.d_model)

    def call(self, x: tf.Tensor, encoding: tf.Tensor) -> tf.Tensor:
        # shape of K, Q, V: (batch_size, n, d_model)
        assert tf.shape(K) == tf.shape(Q) == tf.shape(V), "K, Q, V must have the same shape" # Idk if tf.shape will work with XLA
        # TODO: find another way to determine shape other than passing batch size on model init
        batch_size, n, d_k = tf.shape(K).numpy()
        K, V = self.w_k(encoding), self.w_v(encoding)
        Q = self.w_q(x)

        K = reshape_for_multihead_attention(K, batch_size, n, self.n_heads, self.d_k)  
        Q = reshape_for_multihead_attention(Q, batch_size, n, self.n_heads, self.d_k)
        V = reshape_for_multihead_attention(V, batch_size, n, self.n_heads, self.d_k)
        
        attention_output = self.self_attention(K, Q, V)
        # Concat heads with transpose and reshape
        attention_output = tf.transpose(attention_output, perm=[0, 2, 1, 3]) # shape (batch_size, n, n_heads, d_k)
        attention_output = tf.reshape(attention_output, shape=[batch_size, n, self.d_model]) # shape (batch_size, n, d_model)
        outputs = self.w_o(attention_output)

        return outputs

def relative_attention_buckets(relative_position: tf.Tensor, bidirectional=True, num_buckets=32, max_distance=128) -> tf.Tensor:
    relative_buckets = 0
    if bidirectional:
        num_buckets //= 2
        relative_buckets = num_buckets if relative_position > 0 else 0
        relative_position = tf.math.abs(relative_position)
    else:
        relative_position = -tf.math.minimum(relative_position, tf.zeros_like(relative_position))
    # now relative_position is in the range [0, inf)

    # half of the buckets are for exact increments in positions
    max_exact = num_buckets // 2
    is_small = relative_position < max_exact

    # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
    relative_position_if_large = tf.cast(max_exact + (
        tf.math.log(tf.cast(relative_position, dtype=tf.float32) / max_exact)
        / math.log(max_distance / max_exact)
        * (num_buckets - max_exact)
    ), dtype=tf.int32)
    relative_position_if_large = tf.math.minimum(
        relative_position_if_large, tf.fill(tf.shape(relative_position_if_large), num_buckets - 1)
    )

    relative_buckets += relative_position if is_small else relative_position_if_large
    return relative_buckets