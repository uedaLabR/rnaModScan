"""Keras site classifier with independent reference and per-read encoders."""
from dataclasses import dataclass

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from utils.nn.inputs import CHANNEL_NAMES

MODEL_INPUT_SCHEMA = 'sequence_reads_repeat18_v1'


@dataclass(frozen=True)
class ModelConfig:
    target: str = 'adenine'
    window: int = 19
    read_channels: int = len(CHANNEL_NAMES)
    max_reads: int = 300
    sampling_seed: int = 42
    sequence_embedding: int = 16
    attention_heads: int = 2
    transformer_ff: int = 32
    read_hidden_dim: int = 16
    read_attention_heads: int = 2
    read_transformer_ff: int = 32
    read_embedding: int = 64
    dropout: float = 0.2

    def __post_init__(self):
        if type(self.max_reads) is not int or self.max_reads < 1:
            raise ValueError('max_reads must be a positive integer')
        if self.target != 'adenine' or self.window < 9 or self.window % 2 != 1:
            raise ValueError('Use the joint adenine target and an odd window >=9')
        if min(self.read_channels, self.sequence_embedding, self.read_embedding) < 1:
            raise ValueError('Model sizes must be positive')
        if not 0 <= self.dropout < 1:
            raise ValueError('Invalid dropout')
        if (self.attention_heads < 1 or self.sequence_embedding % self.attention_heads
                or self.transformer_ff < 1):
            raise ValueError('Invalid Transformer/head dimensions')
        if (self.read_hidden_dim < 1 or self.read_attention_heads < 1
                or self.read_hidden_dim % self.read_attention_heads or self.read_transformer_ff < 1):
            raise ValueError('Invalid read Transformer/head dimensions')


@keras.utils.register_keras_serializable(package='rnaModScan')
class TrimerPositionEmbedding(layers.Layer):
    def __init__(self, length, embedding_dim, **kwargs):
        super().__init__(**kwargs)
        self.length, self.embedding_dim = length, embedding_dim
        self.tokens = layers.Embedding(65, embedding_dim, name='trimer_tokens')
        self.positions = layers.Embedding(length, embedding_dim, name='trimer_positions')

    def build(self, input_shape):
        # Direct child build() calls need their own scopes in Keras 2; otherwise
        # both variables are named embeddings:0 and HDF5 checkpointing fails.
        with tf.name_scope(self.tokens.name):
            self.tokens.build(input_shape)
        with tf.name_scope(self.positions.name):
            self.positions.build((self.length,))
        super().build(input_shape)

    def call(self, tokens):
        return self.tokens(tokens) + self.positions(tf.range(self.length))

    def get_config(self):
        return dict(super().get_config(), length=self.length, embedding_dim=self.embedding_dim)


@keras.utils.register_keras_serializable(package='rnaModScan')
class ReadPositionEmbedding(layers.Layer):
    """Add relative window positions to the convolution's per-base features."""
    def __init__(self, length, embedding_dim, **kwargs):
        super().__init__(**kwargs)
        self.length, self.embedding_dim = length, embedding_dim
        self.positions = layers.Embedding(length, embedding_dim, name='read_positions')

    def build(self, input_shape):
        with tf.name_scope(self.positions.name):
            self.positions.build((self.length,))
        super().build(input_shape)

    def call(self, features):
        return features + self.positions(tf.range(self.length))

    def get_config(self):
        return dict(super().get_config(), length=self.length, embedding_dim=self.embedding_dim)


@keras.utils.register_keras_serializable(package='rnaModScan')
class TrimerRepeatEmbedding(layers.Layer):
    """Project the three ordered binary repeat flags of each reference 3-mer."""
    def __init__(self, embedding_dim, **kwargs):
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        # Start from the reference-sequence model and learn repeat contributions.
        self.projection = layers.Dense(embedding_dim, use_bias=False, kernel_initializer='zeros',
                                        name='repeat_binary_projection')

    def build(self, input_shape):
        with tf.name_scope(self.projection.name):
            self.projection.build((input_shape[0], input_shape[1] - 2, 3))
        super().build(input_shape)

    def call(self, inputs):
        binary = tf.cast(inputs, self.compute_dtype)
        triples = tf.stack([binary[:, :-2], binary[:, 1:-1], binary[:, 2:]], axis=-1)
        return self.projection(triples)

    def get_config(self):
        return dict(super().get_config(), embedding_dim=self.embedding_dim)


def _center_and_mean(x, length, name):
    center = length // 2
    middle = layers.Cropping1D((center, length - center - 1), name=name + '_center_crop')(x)
    middle = layers.Flatten(name=name + '_center')(middle)
    mean = layers.GlobalAveragePooling1D(name=name + '_spatial_mean')(x)
    return layers.Concatenate(name=name + '_center_and_context')([middle, mean])


def _read_encoder(window, channels, read_embedding, hidden_dim, attention_heads, transformer_ff):
    features = keras.Input((window, channels), name='one_read_features')
    # Valid 3-base convolution centers positions 1..W-2.
    x = layers.Conv1D(hidden_dim, 3, padding='valid', activation='relu', name='read_local3')(features)
    x = ReadPositionEmbedding(window - 2, hidden_dim, name='read_position_embedding')(x)
    # Self-attention relates nucleotide positions within ONE read. Reads do not
    # attend to each other here, and the reference branch remains independent.
    attended = layers.MultiHeadAttention(num_heads=attention_heads, key_dim=hidden_dim // attention_heads,
                                          name='read_self_attention')(x, x)
    x = layers.LayerNormalization(name='read_attention_norm')(
        layers.Add(name='read_attention_residual')([x, attended]))
    feedforward = layers.Dense(transformer_ff, activation='relu', name='read_ff_hidden')(x)
    feedforward = layers.Dense(hidden_dim, name='read_ff_output')(feedforward)
    x = layers.LayerNormalization(name='read_ff_norm')(
        layers.Add(name='read_ff_residual')([x, feedforward]))
    x = _center_and_mean(x, window - 2, 'read')
    x = layers.Dense(read_embedding, activation='relu', name='read_vector')(x)
    # Reads are independent; padding never enters this shared encoder.
    return keras.Model(features, x, name='shared_read_conv_transformer')


@keras.utils.register_keras_serializable(package='rnaModScan')
class SharedReadEncoder(layers.Layer):
    """Run all real reads through one shared Conv1D/Transformer encoder call."""
    def __init__(self, window, channels, read_embedding=64, hidden_dim=16,
                 attention_heads=2, transformer_ff=32, **kwargs):
        super().__init__(**kwargs)
        self.window, self.channels = window, channels
        self.read_embedding = read_embedding
        self.hidden_dim, self.attention_heads, self.transformer_ff = hidden_dim, attention_heads, transformer_ff
        self.encoder = _read_encoder(window, channels, read_embedding, hidden_dim, attention_heads, transformer_ff)

    def call(self, inputs, training=None):
        features, read_mask = inputs
        read_mask = tf.cast(read_mask, tf.bool)
        tf.debugging.assert_positive(tf.reduce_sum(tf.cast(read_mask, tf.int32), axis=1),
                                     message='Each site needs at least one observed read')
        indices = tf.where(read_mask)
        real_reads = tf.gather_nd(features, indices)
        embeddings = self.encoder(real_reads, training=training)
        shape = tf.concat([tf.shape(read_mask, out_type=tf.int64), [self.read_embedding]], axis=0)
        return tf.scatter_nd(indices, embeddings, shape)

    def compute_output_shape(self, input_shape):
        return (*input_shape[0][:2], self.read_embedding)

    def get_config(self):
        return dict(super().get_config(), window=self.window, channels=self.channels,
                    read_embedding=self.read_embedding, hidden_dim=self.hidden_dim,
                    attention_heads=self.attention_heads, transformer_ff=self.transformer_ff)


@keras.utils.register_keras_serializable(package='rnaModScan')
class MaskedReadAttention(layers.Layer):
    """One softmax over the stored real reads of a site; padding is excluded."""
    def __init__(self, hidden_units=32, **kwargs):
        super().__init__(**kwargs)
        self.hidden_units = hidden_units
        self.hidden = layers.Dense(hidden_units, activation='tanh', name='attention_hidden')
        self.score = layers.Dense(1, use_bias=False, name='attention_score')

    def build(self, input_shape):
        embedding_shape = input_shape[0]
        with tf.name_scope(self.hidden.name):
            self.hidden.build(embedding_shape)
        with tf.name_scope(self.score.name):
            self.score.build((*embedding_shape[:-1], self.hidden_units))
        super().build(input_shape)

    def call(self, inputs):
        embeddings, mask = inputs
        mask = tf.cast(mask, tf.bool)
        logits = tf.squeeze(self.score(self.hidden(embeddings)), axis=-1)
        masked = tf.where(mask, logits, tf.cast(-1e9, logits.dtype))
        weights = tf.nn.softmax(masked, axis=1)
        weights = tf.where(mask, weights, tf.zeros_like(weights))
        weights = tf.math.divide_no_nan(weights, tf.reduce_sum(weights, axis=1, keepdims=True))
        pooled = tf.reduce_sum(embeddings * weights[..., None], axis=1)
        return pooled, weights

    def get_config(self):
        return dict(super().get_config(), hidden_units=self.hidden_units)


def build_site_model(config=None):
    """Return the m6A/inosine/false-positive three-class softmax model.

    Each site has fixed read slots and a separate mask for real reads.
    No WT/IVT identity, genomic IDs or truth/IVT comparisons enter.
    """
    config = config or ModelConfig()
    length = config.window - 2
    inputs = dict(
        sequence_tokens=keras.Input((length,), dtype='int32', name='sequence_tokens'),
        sequence_repeat_mask=keras.Input((config.window,), dtype='bool', name='sequence_repeat_mask'),
        read_features=keras.Input((config.max_reads, config.window, config.read_channels), name='read_features'),
        read_mask=keras.Input((config.max_reads,), dtype='bool', name='read_mask'),
    )
    # 0..63 canonical 3-mers; 64 explicitly unknown (e.g. N), never encoded as AAA.
    sequence = TrimerPositionEmbedding(length, config.sequence_embedding,
                                        name='trimer_position_embedding')(inputs['sequence_tokens'])
    repeat = TrimerRepeatEmbedding(config.sequence_embedding,
                                    name='trimer_repeat_embedding')(inputs['sequence_repeat_mask'])
    sequence = layers.Add(name='sequence_with_repeat')([sequence, repeat])
    attended = layers.MultiHeadAttention(num_heads=config.attention_heads,
                key_dim=config.sequence_embedding // config.attention_heads,
                name='sequence_self_attention')(sequence, sequence)
    sequence = layers.LayerNormalization(name='sequence_attention_norm')(
        layers.Add()([sequence, attended]))
    feedforward = layers.Dense(config.transformer_ff, activation='relu', name='sequence_ff_hidden')(sequence)
    feedforward = layers.Dense(config.sequence_embedding, name='sequence_ff_output')(feedforward)
    sequence = layers.LayerNormalization(name='sequence_ff_norm')(layers.Add()([sequence, feedforward]))
    s = _center_and_mean(sequence, length, 'sequence')
    s = layers.Dense(64, activation='relu', name='sequence_vector')(s)

    reads = SharedReadEncoder(config.window, config.read_channels,
                              config.read_embedding, hidden_dim=config.read_hidden_dim,
                              attention_heads=config.read_attention_heads,
                              transformer_ff=config.read_transformer_ff, name='shared_read_encoder')(
                                    [inputs['read_features'], inputs['read_mask']])
    r, _ = MaskedReadAttention(name='read_attention')([reads, inputs['read_mask']])

    x = layers.Concatenate(name='site_fusion')([r, s])
    x = layers.Dense(64, activation='relu', name='site_dense64')(x)
    x = layers.Dropout(config.dropout, name='site_dropout')(x)
    x = layers.Dense(32, activation='relu', name='site_dense32')(x)
    score = layers.Dense(3, activation='softmax', name='class_scores')(x)
    return keras.Model(inputs, score, name='rnamodscan_adenine')


def compile_site_model(model, learning_rate=1e-3):
    model.compile(optimizer=keras.optimizers.Adam(learning_rate),
                  loss=keras.losses.SparseCategoricalCrossentropy(),
                  metrics=[keras.metrics.SparseCategoricalAccuracy(name='accuracy')])
    return model


def attention_model(model):
    """Diagnostic weights are contributions, not per-read modification labels."""
    return keras.Model(model.inputs, model.get_layer('read_attention').output[1], name=model.name + '_attention')
