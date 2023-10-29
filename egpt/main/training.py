from dataclasses import dataclass, field
import tensorflow as tf
import numpy as np
import keras_nlp
from transformers import DataCollatorWithPadding
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from tensorflow.keras import layers
from tensorflow import keras
from tqdm import tqdm
import pickle as pkl
import transformers


@dataclass(frozen=False, unsafe_hash=True)
class Training:
    num_epochs: int = field(init=True, default=int, repr=False, compare=False)

    def training(self) -> None:
        sampler = keras_nlp.samplers.TopPSampler(p=0.75)
        loss_gpt = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        loss_class = tf.keras.losses.BinaryCrossentropy()
        opt_gpt = tf.keras.optimizers.legacy.Adam(learning_rate=1e-2)
        opt_class = tf.keras.optimizers.legacy.Adam(learning_rate=1e-3)
