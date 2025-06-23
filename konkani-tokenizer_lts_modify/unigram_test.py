import sentencepiece as spm

# Load the trained model
sp = spm.SentencePieceProcessor()
sp.load("spm_konkani.model")

# Test encoding
text = "दुख"  # sample Konkani input
tokens = sp.encode(text, out_type=str)
ids = sp.encode(text, out_type=int)

print("Tokens:", tokens)
print("IDs:", ids)

# Test decoding from ids
decoded = sp.decode(ids)
print("Decoded:", decoded)

# Reverse from tokens (if needed)
ids_from_tokens = [sp.piece_to_id(token) for token in tokens]
print("IDs from Tokens:", ids_from_tokens)
