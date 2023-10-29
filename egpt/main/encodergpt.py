from dataclasses import dataclass, field
from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf
import keras_nlp

@dataclass(frozen=False, unsafe_hash=True)
class EncoderDecoder:
    @staticmethod
    def indepEncoderDecoderSeq(vocab_size: int, seq_len: int, embed_dim: int,
                               num_layers: int, num_heads: int, feed_foward_dim: int) -> tuple:
        # Setting up GPT Model.
        inputs_gpt = keras.layers.Input(shape=(None,), dtype=tf.int32)

        # Shared Embedding.
        shared_embedding_layer = keras_nlp.layers.TokenAndPositionEmbedding(
            vocabulary_size=vocab_size,
            sequence_length=seq_len,
            embedding_dim=embed_dim,
            mask_zero=True,
        )

        x_gpt = shared_embedding_layer(inputs_gpt)

        # Transformer decoders.
        for _ in range(num_layers):
            decoder_layer = keras_nlp.layers.TransformerDecoder(
                num_heads=num_heads,
                intermediate_dim=feed_foward_dim,
            )
            x_gpt = decoder_layer(x_gpt)

        # Output.
        outputs_gpt = keras.layers.Dense(vocab_size)(x_gpt)
        model_gpt = keras.Model(inputs=inputs_gpt, outputs=outputs_gpt)

        # Setting up Encoder Classifier.
        inputs_class = layers.Input(shape=(seq_len,))

        x_class = shared_embedding_layer(inputs_class)

        # Transformer encoders.
        for _ in range(num_layers):
            encoder_layer = keras_nlp.layers.TransformerEncoder(
                num_heads=num_heads,
                intermediate_dim=feed_foward_dim,
            )
            x_class = encoder_layer(x_class)

        x_class = layers.GlobalAveragePooling1D()(x_class)
        x_class = layers.Dropout(0.1)(x_class)
        x_class = layers.Dense(20, activation="relu")(x_class)
        outputs_class = layers.Dense(1, activation="sigmoid")(x_class)
        model_class = keras.Model(inputs=inputs_class, outputs=outputs_class)
        return model_gpt, model_class