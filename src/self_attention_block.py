# self_attention_block.py

import tensorflow as tf
from tensorflow.keras import layers, Model, saving

@tf.keras.saving.register_keras_serializable()
class SelfAttentionBlock(layers.Layer):
    def __init__(self, use_projection=True, **kwargs):
        """
        Слой самовнимания для временных рядов и CNN/MLP/LSTM
        :param use_projection: использовать Dense-проекции для Q/K/V
        """
        super(SelfAttentionBlock, self).__init__(**kwargs)
        self.use_projection = use_projection
        self.feature_dim = None
        self.scale = None
        self.supports_masking = True  # Поддержка LSTM и GRU маскировки

    def build(self, input_shape):
        """
        input_shape == (batch_size, timesteps, features)
        """
        feature_dim = input_shape[-1]

        if self.use_projection:
            self.query_proj = layers.Dense(feature_dim)
            self.key_proj = layers.Dense(feature_dim)
            self.value_proj = layers.Dense(feature_dim)
        else:
            self.query_proj = self.key_proj = self.value_proj = lambda x: x

        # Создаём слои нормализации и суммы
        self.add_layer = layers.Add()
        self.norm_layer = layers.LayerNormalization()

        # Определяем scale
        self.feature_dim = feature_dim
        self.scale = tf.sqrt(tf.cast(feature_dim, tf.float32))

        super().build(input_shape)

    def call(self, inputs):
        """
        inputs: shape == (batch_size, timesteps, features)
        returns: output tensor of the same shape
        """
        query = self.query_proj(inputs)
        key = self.key_proj(inputs)
        value = self.value_proj(inputs)

        scores = tf.matmul(query, key, transpose_b=True) / self.scale
        attention_weights = tf.nn.softmax(scores, axis=-1)
        context = tf.matmul(attention_weights, value)

        # Эти слои уже созданы в build()
        output = self.add_layer([context, inputs])
        output = self.norm_layer(output)

        return output

    def get_config(self):
        config = super(SelfAttentionBlock, self).get_config()
        config.update({
            'use_projection': self.use_projection
        })
        return config