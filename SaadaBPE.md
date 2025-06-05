# SaadaBPE Byte-Pair Encoding (BPE) Tokenizer

A pure Python class for building and using a BPE tokenizer with features tailored for Arabic text normalization and flexible vocabulary management.

---

## Table of Contents

* [Overview](#overview)
* [Installation](#installation)
* [Key Features](#key-features)
* [Parameters](#parameters)
* [Usage Example](#usage-example)
* [API and Methods](#api-and-methods)
* [Model Persistence](#model-persistence)
* [Notes](#notes)

---

## Overview

`SaadaBPE` is a Byte-Pair Encoding (BPE) tokenizer implemented in Python. It supports training, encoding, decoding, saving, and loading. It is well-suited for both general and Arabic text (with optional diacritic handling and custom normalization).

---

## Installation

No external dependencies except Python 3.x (requires only the standard library).

---

## Key Features

* **Arabic text normalization** (with/without diacritics)
* **Customizable space symbol** for word separation
* **BPE vocabulary building and merging**
* **Encoding/decoding to tokens or IDs**
* **Vocabulary and merge rules export/import** (JSON/TXT)
* **Special tokens support** (e.g., <pad>, <unk>, etc.)

---

## Parameters

| Parameter        | Type | Default | Description                                            |
| ---------------- | ---- | ------- | ------------------------------------------------------ |
| space\_symbol    | str  | "▁"     | Symbol used to mark the start of a word                |
| keep\_diacritics | bool | False   | Whether to keep Arabic diacritics during normalization |

---

## Usage Example

### 1. Training a Tokenizer

```python
from FemtoGpt import SaadaBPE  # Adjust import path as needed

bpe = SaadaBPE(space_symbol="▁", keep_diacritics=False)
bpe.read_text("train_corpus.txt")         # Load training text
bpe.split_to_word_chars()                 # Prepare initial vocabulary
bpe.train(num_merges=200, min_frequency=2, special_tokens=["<pad>", "<unk>"])

# Save the trained tokenizer
bpe.save_model("my_tokenizer")
```

### 2. Loading a Tokenizer

```python
bpe = SaadaBPE()
bpe.load_model("my_tokenizer")
```

### 3. Encoding/Decoding

```python
# Encode (get tokens and token ids)
tokens, ids = bpe.encode("مثال عربي للاختبار", return_ids=True)
print(tokens)  # ["▁م", "ث", ...]
print(ids)     # [101, 32, ...]

# Decode from tokens
sentence = bpe.decode(tokens)
print(sentence)

# Decode from IDs
sentence2 = bpe.decode_from_ids(ids)
print(sentence2)
```

### 4. Accessing Vocab and Utility Methods

```python
vocab = bpe.get_vocab()                    # token => id dict
print(len(bpe))                            # vocab size
print(bpe.token_to_id("▁م"))              # get id for token
print(bpe.id_to_token(101))                # get token for id
bpe.print_vocab()                          # print vocab
```

---

## API and Methods

### Initialization

```python
SaadaBPE(space_symbol="▁", keep_diacritics=False)
```

### Training

```python
bpe.read_text(filepath)            # Load raw text
bpe.split_to_word_chars()          # Initialize vocabulary
bpe.train(num_merges=100)          # Run BPE merges
```

### Encoding/Decoding

```python
bpe.encode(text, return_ids=False)          # Encode to tokens/ids
bpe.decode(tokens)                         # Decode tokens to string
bpe.decode_from_ids(ids)                   # Decode ids to string
```

### Vocabulary/ID Utilities

```python
bpe.get_vocab()                            # Get {token: id} dict
bpe.token_to_id(token)                     # Token to id
bpe.id_to_token(idx)                       # Id to token
bpe.convert_tokens_to_ids(tokens)          # List of tokens to ids
bpe.convert_ids_to_tokens(ids)             # List of ids to tokens
len(bpe)                                   # Vocab size
bpe.print_vocab()                          # Print vocab
```

### Saving/Loading Model

```python
bpe.save_model(output_dir="tokenizer_output")   # Save merges/vocab/config
bpe.load_model(output_dir="tokenizer_output")   # Load merges/vocab/config
```

---

## Model Persistence

* Saves `merges.txt`, `vocab.txt`, `config.json`, and `tokenizer.json` in the output directory.
* These files contain all information needed to reload the tokenizer and reproduce results.

---

## Notes

* BPE training must be run after loading and splitting text.
* Use `special_tokens` in `train()` for unknown/padding etc.
* Custom `space_symbol` can help with different language processing scenarios.
* Designed for research, NLP prototyping, and production data pipelines.

---

