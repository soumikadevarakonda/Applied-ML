import contractions
import unicodedata
import re
from word2number import w2n

def remove_accents(text):
    normalized = unicodedata.normalize('NFKD', text)
    return normalized.encode('ASCII', 'ignore').decode('utf-8')

def convert_text_numbers(text):
    def replace_text_num(match):
        try:
            num = w2n.word_to_num(match.group())
            return str(num)
        except:
            return match.group()
    return re.sub(r'\b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|'
                  r'eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|'
                  r'eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|'
                  r'eighty|ninety|hundred|thousand|million|billion)\b(?:\s+\w+){0,3}',
                  replace_text_num, text, flags=re.IGNORECASE)

def normalize_text_pipeline(text):
    text = contractions.fix(text)
    text = remove_accents(text)
    text = convert_text_numbers(text)
    return text

raw_text = "I've got twenty five apples from José's café. He'll give me one hundred more!"
normalized = normalize_text_pipeline(raw_text)

print("Original Text:")
print(raw_text)

print("\nNormalized Text:")
print(normalized)
