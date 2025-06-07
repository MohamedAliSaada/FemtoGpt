import tensorflow as tf

class InputEmbedding(tf.keras.layers.Layer):
    """
    Token + positional embedding + dropout for GPT-style models,
    with full parameter validation.
    """
    def __init__(self, vocab_size, d_model, max_len, dropout_rate=0.1):
        super().__init__()
        # Validate dropout_rate
        assert 0.0 <= dropout_rate < 1.0, f"dropout_rate must be in [0.0, 1.0), got {dropout_rate}"
        # Validate vocab_size, d_model, max_len
        assert vocab_size > 0, f"vocab_size must be positive, got {vocab_size}"
        assert d_model > 0, f"d_model must be positive, got {d_model}"
        assert max_len > 0, f"max_len must be positive, got {max_len}"

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_len = max_len
        self.dropout_rate = dropout_rate

        self.token_emb = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=d_model)
        self.pos_emb = tf.keras.layers.Embedding(input_dim=max_len, output_dim=d_model)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, input_ids, training=None):
        # input_ids: (batch_size, seq_len)
        batch_size = tf.shape(input_ids)[0]
        seq_length = tf.shape(input_ids)[1]

        # Check: sequence length must not exceed max_len
        tf.debugging.assert_less_equal(
            seq_length,
            self.max_len,
            message=f"Sequence length must be ≤ {self.max_len}"
        )

        # Check: all token ids must be < vocab_size
        tf.debugging.assert_less(
            tf.reduce_max(input_ids),
            self.vocab_size,
            message=f"All input IDs must be < vocab_size ({self.vocab_size})"
        )

        # Prepare position ids
        position_ids = tf.tile(
            tf.range(seq_length)[tf.newaxis, :],
            [batch_size, 1]
        )  # shape: (batch_size, seq_len)

        # Embedding lookup
        token_embeddings = self.token_emb(input_ids)      # (batch_size, seq_len, d_model)
        pos_embeddings = self.pos_emb(position_ids)       # (batch_size, seq_len, d_model)

        # Check: final embedding shape (for debug)
        tf.debugging.assert_equal(
            tf.shape(token_embeddings)[-1],
            self.d_model,
            message=f"Token embedding last dim must be {self.d_model}"
        )

        x = token_embeddings + pos_embeddings
        x = self.dropout(x, training=training)
        return x
