# ASR Fine-Tuning Using LLMs (Konkani Language)

This repository focuses on fine-tuning Automatic Speech Recognition (ASR) models for the Konkani language using custom-built tokenizers and domain-specific datasets. It includes training pipelines, tokenizer experiments, and preprocessing utilities to enable effective ASR with support for multiple tokenizer types (Unigram, BPE, WordPiece).

---

## 🧠 Project Highlights

- Custom SentencePiece-based tokenizer for Konkani (Unigram, BPE, WordPiece)
- Trained on textbook corpora and real-world audio transcripts
- Integrated with ASR model fine-tuning pipelines
- Includes multiple training strategies and tokenizer evaluations

---



## 🔧 Requirements

```bash
pip install sentencepiece
```

🔁 Training a Tokenizer

Inside konkani-tokenizer_lts_modify/tokenizer.ipynb, training is done using:

```bash
import sentencepiece as spm

spm.SentencePieceTrainer.train(
    input=['corpus.txt', 'all_transcripts.txt', 'konkani_clean.txt'],
    model_prefix='spm_konkani',
    vocab_size=8000,
    model_type='unigram',
    character_coverage=1.0,
    user_defined_symbols=["<mask>"]
)

```

```bash
🔍 Usage Example

import sentencepiece as spm

sp = spm.SentencePieceProcessor(model_file="spm_konkani.model")

text = "तुमका किते कयलो?"
ids = sp.encode(text, out_type=int)
tokens = sp.encode(text, out_type=str)
print("Tokens:", tokens)
print("Decoded:", sp.decode(ids))

```