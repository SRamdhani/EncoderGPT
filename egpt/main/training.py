from dataclasses import dataclass, field
from ..utils.util import Utility
import tensorflow as tf
import numpy as np
import keras_nlp

@dataclass(frozen=False, unsafe_hash=True)
class Training(Utility):
    num_epochs: int = field(init=True, default=int, repr=False, compare=False)

    def next_gpt(self, prompt, cache, index):
        logits = self.model_gpt(prompt)[:, index - 1, :]
        return logits, None, cache

    def training(self, train_data: tf.Tensor, test_data: tf.Tensor,
                 epochs: int = 1, step_check: int = 1000) -> None:

        sampler = keras_nlp.samplers.TopPSampler(p=0.75)
        loss_gpt = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        loss_class = tf.keras.losses.BinaryCrossentropy()
        opt_gpt = tf.keras.optimizers.legacy.Adam(learning_rate=1e-2)
        opt_class = tf.keras.optimizers.legacy.Adam(learning_rate=1e-3)

        for epoch in range(epochs):
            metric = tf.keras.metrics.F1Score(threshold=0.5)
            total_batches = len(train_data)

            for step, x in enumerate(train_data):

                with tf.GradientTape(persistent=True) as tape:
                    output_gpt = self.model_gpt(x['input_ids'])
                    outputs_class = []

                    if step % step_check == 0:
                        for b in range(self.batch_size):
                            if 102 in x['labels_class'][b, :]:
                                sampled = sampler(
                                    next=self.next_gpt,
                                    prompt=tf.expand_dims(x['labels_class'][b, :], 0),
                                    index=np.where(x['labels_class'][b, :] == 102)[0][0]
                                )
                                outputs_class.append(sampled)
                            else:
                                outputs_class.append(x['labels_class'][b, :])
                    else:
                        outputs_class = x['labels_class']

                    input_classifier = tf.squeeze(tf.stack(outputs_class))
                    output_class = self.model_class(input_classifier)

                    generative_loss = loss_gpt(x['labels_gpt'], output_gpt)
                    classifier_loss = loss_class(x['label'], output_class)

                gpt_grads = tape.gradient(generative_loss, self.model_gpt.trainable_weights)
                gpt_trainables = self.model_gpt.trainable_weights

                class_grads = tape.gradient(classifier_loss, self.model_class.trainable_weights)
                class_trainables = self.model_class.trainable_weights

                opt_gpt.apply_gradients(zip(gpt_grads, gpt_trainables))
                opt_class.apply_gradients(zip(class_grads, class_trainables))
                metric.update_state(tf.reshape(tf.cast(x['label'], tf.float32), [-1, 1]), output_class)

                if step % step_check == 0:
                    print('Epoch: {} at step: {} and remaining steps: {}'.format(epoch, step, total_batches - step))
                    print('Generative Loss: {}, Classifier Loss: {} and F1: {}'.format(generative_loss.numpy(),
                                                                                       classifier_loss.numpy(),
                                                                                       metric.result().numpy()[-1]))
                    print()

            print('Final Averaged F1: ', metric.result().numpy().mean())