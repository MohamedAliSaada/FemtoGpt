# SentencePiece Tokenizer: Quickstart

This guide shows how to train a SentencePiece tokenizer, what files you get, and how to use your trained tokenizer in Python.

---

## 1. How to Train a Tokenizer

Install SentencePiece:

```bash
pip install sentencepiece
```

Train on your text data:

```python
import sentencepiece as spm

spm.SentencePieceTrainer.train(
    input='data.txt',         # Path to your training text file
    model_prefix='mymodel',   # Prefix for output files
    vocab_size=8000,          # Number of tokens in the vocab
    model_type='bpe'          # Model type: bpe/unigram/char/word
)
```

---

## 2. Output Files

After training, you’ll get:

* `mymodel.model`    → The trained SentencePiece model (binary)
* `mymodel.vocab`    → The vocabulary file (list of tokens, text format)

---

## 3. How to Use Your Trained Tokenizer

Load and use your model in Python:

```python
import sentencepiece as spm

sp = spm.SentencePieceProcessor()
sp.load('mymodel.model')

# Encode text to IDs
ids = sp.encode('This is a test.', out_type=int)
print('Token IDs:', ids)

# Decode IDs back to text
text = sp.decode(ids)
print('Decoded text:', text)

# Encode text to tokens (subwords)
tokens = sp.encode('This is a test.', out_type=str)
print('Tokens:', tokens)
```

---

## References

* [SentencePiece GitHub](https://github.com/google/sentencepiece)
* [Python API Docs](https://github.com/google/sentencepiece/blob/master/python/README.md)




