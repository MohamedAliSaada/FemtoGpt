import tensorflow as tf
import numpy as np

class Trainer:
    def __init__(self, model, optimizer=None, loss_fn=None, batch_size=32):
        self.model = model
        self.batch_size = batch_size

        self.optimizer = optimizer or tf.keras.optimizers.Adam(learning_rate=1e-4)
        self.loss_fn = loss_fn or tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    def train(self, dataset, epochs=3):
        for epoch in range(epochs):
            total_loss = 0.0
            total_accuracy = 0.0
            steps = 0

            for batch in dataset:
                x_batch, y_batch = batch
                with tf.GradientTape() as tape:
                    logits = self.model(x_batch, training=True)
                    loss = self.loss_fn(y_batch, logits)

                grads = tape.gradient(loss, self.model.trainable_variables)
                self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

                predictions = tf.argmax(logits, axis=-1)
                y_batch = tf.cast(y_batch, tf.int64)
                mask = tf.cast(tf.not_equal(y_batch, 0), dtype=tf.float32)
                correct = tf.cast(tf.equal(predictions, y_batch), dtype=tf.float32)
                accuracy = tf.reduce_sum(correct * mask) / tf.reduce_sum(mask)

                total_loss += loss.numpy()
                total_accuracy += accuracy.numpy()
                steps += 1

            perplexity = np.exp(total_loss / steps)

            print(f"Epoch {epoch + 1}: Loss = {total_loss / steps:.4f}, Accuracy = {total_accuracy / steps:.4f}, Perplexity = {perplexity:.4f}")
