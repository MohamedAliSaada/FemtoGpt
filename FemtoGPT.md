# FemtoGPT

A minimal and educational GPT-style language model implementation using TensorFlow/Keras.

---

## 📦 Features

* Custom tokenizer support
* Token + Positional Embedding
* Custom Transformer blocks
* Output head with layer normalization

---

## 🧱 Architecture Overview

```
Input IDs → InputEmbedding → Stack of DTransformer Blocks → OutputHead → Logits
```

---

## 🔧 Installation

You can use FemtoGPT directly in Colab using:

```python
!pip install git+https://github.com/MohamedAliSaada/FemtoGpt.git
```

---

## 🚀 How to Use

```python
from FemtoGpt import InputEmbedding, DTransformer, OutputHead, FemtoGPT
import tensorflow as tf

# Step 1: Define Configuration
gpt_config = {
    "vocab_size": 6000,
    "max_len": 512,
    "d_model": 128,
    "dropout_rate": 0.1,
    "num_heads": 2,
    "expansion": 4,
    "num_layers": 2,
    "rich": 0
}

# Step 2: Initialize Model
model = FemtoGPT(gpt_config)

# Step 3: Dummy Input (batch of token IDs)
x = tf.random.uniform((2, 16), dtype=tf.int32, maxval=gpt_config["vocab_size"])
logits = model(x)
print(logits.shape)  # (2, 16, 6000)
```

---

## 📚 Components

* `InputEmbedding`: token + position embeddings + dropout
* `DTransformer`: pre-norm transformer block with MHA + FFN
* `OutputHead`: layer norm + final dense layer
* `FemtoGPT`: full model wrapper

---

## 📌 Notes

* Padding is handled via masking in Transformer layers.
* Outputs include logits for each token in the sequence (including padding positions).

---

## 👤 Author

**Mohamed Ali Saada**
For contributions, open an issue or pull request on [GitHub](https://github.com/MohamedAliSaada/FemtoGpt)

---


