import sentencepiece as spm

# Load the trained model
sp = spm.SentencePieceProcessor()
sp.load("spm_konkani.model")

# Test sentence
text = "हांव कालेर मोरगांव गेलो आंव भितर थोडे वेळाचेर मितरांक सोबत उबोंट आनी चाय घेतलो, तांचे संगत मजेचेर आसा."

# Encode
tokens = sp.encode(text, out_type=str)
ids = sp.encode(text, out_type=int)

print("Tokens:", tokens)
print("IDs:", ids)

# Decode
decoded = sp.decode(ids)
print("Decoded:", decoded)

# Reverse map
ids_from_tokens = [sp.piece_to_id(token) for token in tokens]
print("IDs from Tokens:", ids_from_tokens)

# Character-level accuracy
char_match = sum(1 for a, b in zip(text, decoded) if a == b)
char_accuracy = char_match / max(len(text), len(decoded)) * 100
print(f"Character Accuracy: {char_accuracy:.2f}%")

# Token-level accuracy
re_tokens = sp.encode(decoded, out_type=str)
token_match = sum(1 for a, b in zip(tokens, re_tokens) if a == b)
token_accuracy = token_match / max(len(tokens), len(re_tokens)) * 100
print(f"Token Accuracy: {token_accuracy:.2f}%")
