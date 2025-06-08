import tensorflow as tf

class Trainer:
    def __init__(self, model, optimizer, loss_fn, tokenizer, config):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.tokenizer = tokenizer
        self.config = config

    def _compute_metrics(self, labels, logits):
        predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
        mask = tf.cast(tf.not_equal(labels, 0), tf.float32)
        correct = tf.cast(tf.equal(predictions, labels), tf.float32)
        accuracy = tf.reduce_sum(correct * mask) / tf.reduce_sum(mask)
        loss = self.loss_fn(labels, logits)
        perplexity = tf.exp(loss)
        return loss, accuracy, perplexity

    def train(self, dataset, epochs=1):
        for epoch in range(epochs):
            total_loss = 0.0
            total_accuracy = 0.0
            total_perplexity = 0.0
            steps = 0

            for batch in dataset:
                inputs, labels = batch

                with tf.GradientTape() as tape:
                    logits = self.model(inputs, training=True)
                    loss, accuracy, perplexity = self._compute_metrics(labels, logits)

                grads = tape.gradient(loss, self.model.trainable_variables)
                self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

                total_loss += loss.numpy()
                total_accuracy += accuracy.numpy()
                total_perplexity += perplexity.numpy()
                steps += 1

            print(f"Epoch {epoch + 1}: Loss = {total_loss / steps:.4f} | Accuracy = {total_accuracy / steps:.4f} | Perplexity = {total_perplexity / steps:.4f}")
