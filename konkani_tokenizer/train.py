import sentencepiece as spm

spm.SentencePieceTrainer.Train(
    '--input=konkani_clean.txt --model_prefix=konkani --vocab_size=1000 --character_coverage=1.0 --model_type=bpe'
)
