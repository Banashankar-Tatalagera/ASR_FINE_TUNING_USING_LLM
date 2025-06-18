import re

with open('konkani.pos', 'r', encoding='utf-8') as infile, open('konkani_clean.txt', 'w', encoding='utf-8') as outfile:
    for line in infile:
        # Remove POS tags
        cleaned = ' '.join([re.sub(r'/.*$', '', token) for token in line.strip().split()])
        outfile.write(cleaned + '\n')
