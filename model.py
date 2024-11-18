# model.py
import pickle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.saving import register_keras_serializable


# Registering the custom transformer model
@register_keras_serializable()
class TransformerBlock(layers.Layer):
    def __init__(self, embed_size, heads, dropout_rate=0.1, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.attention = layers.MultiHeadAttention(num_heads=heads, key_dim=embed_size)
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.feed_forward = keras.Sequential([
            layers.Dense(embed_size * 4, activation="relu"),
            layers.Dense(embed_size)
        ])
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)

    def call(self, inputs, training):
        attn_output = self.attention(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.norm1(inputs + attn_output)
        ff_output = self.feed_forward(out1)
        ff_output = self.dropout2(ff_output, training=training)
        return self.norm2(out1 + ff_output)

    def get_config(self):
        config = super(TransformerBlock, self).get_config()
        return config

@register_keras_serializable()
class SimpleTransformer(keras.Model):
    def __init__(self, num_layers, embed_size, num_heads, vocab_size, max_length, dropout_rate=0.1, **kwargs):
        super(SimpleTransformer, self).__init__(**kwargs)
        self.embedding = layers.Embedding(vocab_size, embed_size)
        self.positional_encoding = self.positional_encoding(max_length, embed_size)
        self.transformer_blocks = [TransformerBlock(embed_size, num_heads, dropout_rate) for _ in range(num_layers)]
        self.pooling = layers.GlobalAveragePooling1D()
        self.final_layer = layers.Dense(1, activation='sigmoid')

    def positional_encoding(self, max_length, embed_size):
        position = tf.range(max_length, dtype=tf.float32)[:, tf.newaxis]
        div_term = tf.pow(10000, (2 * (tf.range(embed_size // 2, dtype=tf.float32) // 2)) / embed_size)

        sin_values = tf.sin(position / div_term)
        cos_values = tf.cos(position / div_term)
        pos_enc = tf.concat([sin_values, cos_values], axis=-1)

        return pos_enc[tf.newaxis, ...]

    def call(self, inputs, training=False):
        seq_len = tf.shape(inputs)[1]
        x = self.embedding(inputs) + self.positional_encoding[:, :seq_len, :]
        for transformer in self.transformer_blocks:
            x = transformer(x, training=training)
        x = self.pooling(x)
        return self.final_layer(x)

    def get_config(self):
        config = super(SimpleTransformer, self).get_config()
        config.update({
            'num_layers': self.num_layers,
            'embed_size': self.embed_size,
            'num_heads': self.num_heads,
            'vocab_size': self.vocab_size,
            'max_length': self.max_length,
            'dropout_rate': self.dropout_rate,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

# Function to load the model and tokenizer
def load_model_and_tokenizer():
    # Load the tokenizer
    with open("tokenizer_syn.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    
    # Load the model
    model = keras.models.load_model("xss_detector_model_syn.keras", custom_objects={"SimpleTransformer": SimpleTransformer, "TransformerBlock": TransformerBlock})
    
    return model, tokenizer
