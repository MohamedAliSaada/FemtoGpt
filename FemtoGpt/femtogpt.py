import tensorflow as tf
from .decoder import DTransformer
from .embedding import InputEmbedding
from .outputhead import OutputHead

class FemtoGPT(tf.keras.Model):
    def __init__(self, femto_config):
        super().__init__()

        self.embedding = InputEmbedding(
            vocab_size=femto_config['vocab_size'],
            max_len=femto_config['max_len'],
            d_model=femto_config['d_model'],
            dropout_rate=femto_config['dropout_rate']
        )

        self.transformers = [
            DTransformer(
                embed_dim=femto_config["d_model"],
                num_heads=femto_config["num_heads"],
                expansion=femto_config["expansion"],
                dropout_rate=femto_config["dropout_rate"],
                rich=femto_config.get("rich", 0)
            )
            for _ in range(femto_config['num_layers'])
        ]

        self.output_head = OutputHead(femto_config['vocab_size'])

    def call(self, ids, training=None, mask=None):
        x = self.embedding(ids, training=training)
        for layer in self.transformers:
            x = layer(x, mask=mask, training=training)
        logits = self.output_head(x)
        return logits
