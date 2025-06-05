# Femto6000Tokenizer

**Femto6000Tokenizer** is a Python tokenizer module that wraps a trained [SentencePiece](https://github.com/google/sentencepiece) model for easy and efficient encoding and decoding of Arabic text using subword tokenization. It's built as part of the `FemtoGpt` project to enable quick access to tokenizer functionality with minimal setup.

---

## ✨ Features

* 🧠 Built on [SentencePiece](https://github.com/google/sentencepiece)
* 🔡 Encodes text into subword tokens or IDs
* 🔁 Decodes back to text
* 🧩 Supports vocab lookups: `id_to_piece`, `piece_to_id`
* 🗂️ Loads tokenizer from local `.model` and `.vocab` files included with the package

---

## 📦 Requirements

Install required packages:

```bash
pip install sentencepiece
```

If you're using TensorFlow as part of your GPT model training:

```bash
pip install tensorflow>=2.11
```

Or install everything together via GitHub:

```bash
pip install git+https://github.com/MohamedAliSaada/FemtoGpt.git
```

---

## 📁 File Structure

Your directory should look like this:

```
FemtoGpt/
├── FemtoGpt/
│   ├── __init__.py
│   ├── sp_tokenizer.py         # Contains Femto6000Tokenizer class
│   ├── AR6000SPT.model         # SentencePiece model file
│   └── AR6000SPT.vocab         # Vocabulary file
├── setup.py
├── MANIFEST.in
└── README.md
```

---

## 🛠️ Setup for Local Development

### 1. Clone the Repo

```bash
git clone https://github.com/MohamedAliSaada/FemtoGpt.git
cd FemtoGpt
```

### 2. Install Dependencies

```bash
pip install -e .
```

This will install the package locally, along with its dependencies.

---

## 🚀 How to Use

### 🐍 In Python

```python
from FemtoGpt import Femto6000Tokenizer

tokenizer = Femto6000Tokenizer()

text = "مرحباً بك في فيمتو جي بي تي!"

ids = tokenizer.encode(text, out_type='id')
print("Token IDs:", ids)

tokens = tokenizer.encode(text, out_type='piece')
print("Subwords:", tokens)

decoded = tokenizer.decode(ids)
print("Decoded Text:", decoded)
```

### 📘 In Google Colab

```python
!pip install git+https://github.com/MohamedAliSaada/FemtoGpt.git

from FemtoGpt import Femto6000Tokenizer

tokenizer = Femto6000Tokenizer()
print(tokenizer.encode("مثال على الترميز", out_type='id'))
```

---

## 📜 MANIFEST & Packaging Notes

Ensure your `MANIFEST.in` includes the model files:

```
include FemtoGpt/AR6000SPT.model
include FemtoGpt/AR6000SPT.vocab
```

Also make sure `setup.py` contains:

```python
include_package_data=True
```

This ensures the tokenizer model files are bundled when the package is installed.

---

## ⚠️ Troubleshooting

* `FileNotFoundError`: Make sure `.model` and `.vocab` files exist in the `FemtoGpt/` directory.
* In Colab, ensure files are in the current working directory if not using pip install.

---


