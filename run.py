from transformers import DataCollatorWithPadding
from transformers import AutoTokenizer
from egpt.utils.util import Utility
from egpt.main.model import Model

import tensorflow as tf

# Work around because the Macbook M2 GPU has issue with keras-nlp sampler.
tf.config.set_visible_devices([], 'GPU')

# Define Model Parameters
TOKENIZER         = AutoTokenizer.from_pretrained("distilbert-base-uncased")
VOCAB_SIZE        = TOKENIZER.vocab_size
SEQ_LEN           = 128
EMBED_DIM         = 300
FEED_FORWARD_DIM  = 32
BATCH_SIZE        = 2
NUM_HEADS         = 3
NUM_LAYERS        = 2
EPOCHS            = 3
DATA_COLLATOR     = DataCollatorWithPadding(tokenizer=TOKENIZER,
                                            max_length=SEQ_LEN,
                                            padding='max_length',
                                            return_tensors="tf")

# Warning: Data load may take some time the first time.
u = Utility(seq_len=SEQ_LEN, tokenizer=TOKENIZER, data_collator=DATA_COLLATOR)
training_collated, testing_collated = u.loadDataFromHubOrDisk(dataset="SetFit/amazon_polarity")

# m = Model(num_epochs=EPOCHS, feed_foward_dim=FEED_FORWARD_DIM,
#           vocab_size=VOCAB_SIZE, seq_len=SEQ_LEN, batch_size=BATCH_SIZE,
#           embed_dim=EMBED_DIM, num_heads=NUM_HEADS, num_layers=NUM_LAYERS)
#
# print(m)

