import unicodedata

text = "Café déjà vu! El niño está jugando en la piñata. Pokémon énergies are rare."

def normalize_text(sentence):
    normalized = unicodedata.normalize('NFKD', sentence)
    ascii_text = normalized.encode('ASCII', 'ignore').decode('utf-8')
    return ascii_text

cleaned_text = normalize_text(text)

print("Original Text:")
print(text)

print("\nNormalized Text:")
print(cleaned_text)
