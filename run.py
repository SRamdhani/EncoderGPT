from transformers import DataCollatorWithPadding
from transformers import AutoTokenizer
from egpt.utils.util import Utility
from egpt.main.model import Model
import tensorflow as tf

# TODO: Need to add logging.
# TODO: Need to add validation for testing data

# Work around because the Macbook M2 GPU has issue with keras-nlp sampler.
tf.config.set_visible_devices([], 'GPU')

# Define Model Parameters
TOKENIZER         = AutoTokenizer.from_pretrained("distilbert-base-uncased")
VOCAB_SIZE        = TOKENIZER.vocab_size
SEQ_LEN           = 128
EMBED_DIM         = 300
FEED_FORWARD_DIM  = 32
BATCH_SIZE        = 5
NUM_HEADS         = 3
NUM_LAYERS        = 2
EPOCHS            = 3
DATA_COLLATOR     = DataCollatorWithPadding(tokenizer=TOKENIZER,
                                            max_length=SEQ_LEN,
                                            padding='max_length',
                                            return_tensors="tf")

# # Warning: Data load may take some time the first time.
u = Utility(seq_len=SEQ_LEN, tokenizer=TOKENIZER, data_collator=DATA_COLLATOR)
training_data, testing_data = u.loadDataFromHubOrDisk(dataset="SetFit/amazon_polarity",
                                                      batch_size=BATCH_SIZE)

print(training_data)
# train_data = training_data.to_tf_dataset(
#     columns=['input_ids', 'label', 'labels_gpt', 'labels_class'],
#     batch_size=BATCH_SIZE,
#     shuffle=True)

# Setting up model and training.
# m = Model(num_epochs=EPOCHS, feed_foward_dim=FEED_FORWARD_DIM,
#           vocab_size=VOCAB_SIZE, seq_len=SEQ_LEN, batch_size=BATCH_SIZE,
#           embed_dim=EMBED_DIM, num_heads=NUM_HEADS, num_layers=NUM_LAYERS)
#
# m.training(epochs=1)
# for x in training_data: break
# print(x['input_ids'])
# output_gpt = m.model_gpt(x['input_ids'])
# input_classifier = tf.squeeze(tf.stack(outputs_class))
# output_class = model_class(input_classifier)


