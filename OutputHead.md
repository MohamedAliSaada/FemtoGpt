# FemtoGpt: Build GPT from Scratch with Keras

Welcome to **FemtoGpt**, a minimalist GPT implementation built entirely from scratch using **Keras** and **TensorFlow**. This project aims to help you deeply understand how GPT models work by implementing every key component manually, step by step.

---

## 🚀 Project Structure

| Component   | Description                                               |
| ----------- | --------------------------------------------------------- |
| `FemtoGpt/` | Core module: tokenizer, embedding, transformer, and model |
| `README.md` | Project documentation                                     |

---

## ✅ Completed Modules

### 1. **Tokenizer**

* Uses Google's **SentencePiece** to generate a 6000-token vocabulary.
* Supports subword encoding for efficient language modeling.

### 2. **Input Embedding Layer**

* Token embedding
* Learnable positional embedding
* Dropout for regularization
* Input validation for sequence length and token range

### 3. **Transformer Block (`DTransformer`)**

* Multi-head self-attention
* Layer normalization and residual connections
* Feedforward network

### 4. **Output Head**

* Includes final LayerNorm
* Projects transformer output to vocabulary size logits

---

## 🔧 Coming Soon

* Full GPT model wrapper (`GPTModel`)
* Training loop and loss function
* Inference and text generation scripts

---

## 🛠️ How to Use

### ✅ Recommended (for Google Colab or pip install)

```bash
pip install git+https://github.com/MohamedAliSaada/FemtoGpt.git
```

### 🧪 Local Development

```bash
# Clone the repo
$ git clone https://github.com/MohamedAliSaada/FemtoGpt.git
$ cd FemtoGpt

# Install requirements (if any in future)
$ pip install -r requirements.txt
```

### Example: Use the Input Embedding and Transformer

```python
from FemtoGpt import InputEmbedding, DTransformer, OutputHead

# Config
vocab_size = 6000
max_len = 512
d_model = 128

# Layers
embd = InputEmbedding(vocab_size=vocab_size, max_len=max_len, d_model=d_model, dropout_rate=0.1)
trans = DTransformer(embed_dim=d_model, num_heads=2, expansion=4, dropout_rate=0.1)
head = OutputHead(vocab_size=vocab_size)

# Dummy input
x = tf.random.uniform(shape=(2, 10), maxval=vocab_size, dtype=tf.int32)

# Forward pass
x = embd(x)
x = trans(x)
logits = head(x)
```

---

## 📚 References

* [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
* [GPT Paper](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)
* [TensorFlow Documentation](https://www.tensorflow.org/api_docs)
* [SentencePiece](https://github.com/google/sentencepiece)

---

## 🤝 Contributing

This project is educational, but contributions for improvement are welcome!

* Fork it
* Create your feature branch (`git checkout -b feature/Foo`)
* Commit your changes (`git commit -am 'Add Foo'`)
* Push to the branch (`git push origin feature/Foo`)
* Create a new Pull Request

---

## 📩 Contact

Created with ❤️ by **Mohamed Ali Saada**
Feel free to open issues or suggestions in the repo!

---
