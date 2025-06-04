# DTransformer Layer

A custom TensorFlow Keras layer implementing an encoder block: **Pre-Norm Multi-Head Attention + Feed-Forward Network (FFN)** with residual connections.

---

## Table of Contents

* [Installation](#installation)
* [DTransformer Overview](#dtransformer-overview)
* [Parameters](#parameters)
* [Usage Example](#usage-example)
* [All Methods and Parameters](#all-methods-and-parameters)
* [Error Handling](#error-handling)
* [Notes](#notes)

---

## Installation

```bash
pip install tensorflow
```

---

## DTransformer Overview

The `DTransformer` layer implements a single encoder block consisting of:

* **Layer Normalization** (Pre-Norm)
* **Multi-Head Self-Attention** (`tf.keras.layers.MultiHeadAttention`)
* **Feed-Forward Neural Network** (`Dense -> GELU -> Dense`)
* **Dropout** and **Residual connections** throughout

---

## Parameters

| Parameter     | Type  | Default | Description                                                             |
| ------------- | ----- | ------- | ----------------------------------------------------------------------- |
| embed\_dim    | int   | —       | Input & output embedding dimension (must match input last dimension)    |
| num\_heads    | int   | 12      | Number of attention heads                                               |
| expansion     | int   | 4       | FFN expansion factor (hidden dim = embed\_dim \* expansion)             |
| dropout\_rate | float | 0.1     | Dropout rate for attention and FFN                                      |
| rich          | int   | 0       | Adds extra dimensions to embed\_dim for key/query (advanced, usually 0) |
| \*\*kwargs    | dict  |         | Any extra args for `tf.keras.layers.Layer`                              |

> ⚠️ `(embed_dim + rich)` **must be divisible by** `num_heads`.

---

## Usage Example

```python
import tensorflow as tf
from your_module import DTransformer  # Update import as needed

# Example: Create dummy input
batch_size = 2
seq_len = 8
embed_dim = 32

x = tf.random.uniform((batch_size, seq_len, embed_dim))

# Create a DTransformer block
encoder = DTransformer(
    embed_dim=32,       # Required, must match x.shape[-1]
    num_heads=8,        # Number of attention heads
    expansion=4,        # Feedforward expansion factor
    dropout_rate=0.1,   # Dropout rate
    rich=0              # Optional, advanced usage
)

# Forward pass (no mask)
out = encoder(x, training=True)
print(out.shape)  # (batch_size, seq_len, embed_dim)

# Example with attention mask
mask = tf.ones((batch_size, seq_len), dtype=tf.bool)  # Full attention
out = encoder(x, mask=mask, training=True)
```

---

## All Methods and Parameters

### Initialization

```python
DTransformer(
    embed_dim: int,         # Required: input/output dimension
    num_heads: int = 12,    # Optional: number of attention heads
    expansion: int = 4,     # Optional: FFN expansion
    dropout_rate: float = 0.1,  # Optional: dropout rate
    rich: int = 0,          # Optional: extra embed dims for attention
    **kwargs
)
```

### Forward Pass (`call`)

```python
out = encoder(x, mask=None, training=None)
```

* `x`: Tensor of shape `(batch, seq_len, embed_dim)`
* `mask`: (Optional) Attention mask, typically of shape `(batch, seq_len)` or broadcastable
* `training`: (Optional) Boolean for training/inference mode (dropout control)

---

## Error Handling

* **ValueError**: If `embed_dim + rich` is not divisible by `num_heads`
* **ValueError**: If the input tensor's last dimension does not match `embed_dim`

```python
# Example: Will raise ValueError if input dim is wrong
bad_x = tf.random.uniform((batch_size, seq_len, 30))
out = encoder(bad_x)  # Raises ValueError
```

---

## Notes

* Layer can be stacked to form deeper encoders.
* Designed for use in transformer architectures (ViT, NLP, etc).
* `rich` parameter allows for "extra" dimensions for richer attention (advanced).

---

## Minimal Working Example

```python
import tensorflow as tf

# Create input
x = tf.random.normal((4, 10, 64))

# Instantiate layer
block = DTransformer(embed_dim=64, num_heads=8, expansion=4, dropout_rate=0.1, rich=0)

# Forward
out = block(x, training=True)

print(out.shape)  # (4, 10, 64)
```


