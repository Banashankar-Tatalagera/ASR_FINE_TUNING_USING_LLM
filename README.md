# ASR Fine-Tuning for Konkani using LLM and SentencePiece

This project focuses on **fine-tuning Automatic Speech Recognition (ASR)** models for the **Konkani language**, a low-resource Indian language, using **pretrained models from Hugging Face** and **custom SentencePiece tokenizers**. It aims to improve transcription accuracy by adapting models with a tailored vocabulary and fine-tuning pipeline.

---

## 📌 Key Features

- Fine-tuning of Wav2Vec2 / Whisper / XLS-R models
- Custom tokenizer using SentencePiece (Unigram/BPE)
- Support for noisy or code-switched datasets
- Evaluation on CER/WER metrics
- Supports Hugging Face `transformers` + `datasets` + `evaluate`

---

## 🧩 Dataset Structure

ta/
├── corpus.txt # raw text corpus for tokenizer
├── train.json # ASR training data (path, audio, text)
├── test.json
├── all_transcripts.txt # combined text for vocab building
├── konkani_clean.txt # optional cleaned corpus




---

## 🧠 Tokenizer Training (SentencePiece)

```bash
# Install once
pip install sentencepiece

# Python code
import sentencepiece as spm

spm.SentencePieceTrainer.train(
    input=['corpus.txt', 'all_transcripts.txt', 'konkani_clean.txt'],
    model_prefix='spm_konkani',
    vocab_size=8000,
    model_type='unigram',  # Or 'bpe'
    character_coverage=1.0,
    pad_id=0,
    unk_id=1,
    bos_id=2,
    eos_id=3,
    user_defined_symbols=["<mask>"]
)


python run_asr.py \
  --model_name_or_path facebook/wav2vec2-large-xlsr-53 \
  --dataset_name ./data \
  --tokenizer_path ./spm_konkani.model \
  --output_dir ./results \
  --num_train_epochs 25 \
  --learning_rate 3e-5 \
  --per_device_train_batch_size 16 \
  --fp16 \
  --gradient_accumulation_steps 2 \
  --evaluation_strategy steps \
  --save_steps 100 \
  --eval_steps 100 \
  --logging_steps 10 \
  --do_train \
  --do_eval \
  --predict_with_generate \
  --report_to tensorboard
