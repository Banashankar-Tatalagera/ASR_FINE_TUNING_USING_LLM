import sentencepiece as spm

# Load the tokenizer model
sp = spm.SentencePieceProcessor()
sp.load('konkani.model')

# Take input from the user
user_input = input("Enter a sentence in Konkani: ")

# Encode the input
encoded = sp.encode(user_input, out_type=str)
print('Encoded:', encoded)

# Decode it back to text
decoded = sp.decode(encoded)
print('Decoded:', decoded)
