import re
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Sample documents
documents = [
    "Hello, world! This is a test sentence. Let's see how it works.",
    "Another example sentence to demonstrate text processing.",
    "NLTK is a powerful library for natural language processing.",
    "Text preprocessing is essential for machine learning tasks."
]

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def stem_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words and word not in string.punctuation]
    stemmed = [stemmer.stem(word) for word in tokens]
    return ' '.join(stemmed)

# Apply and print
for i, doc in enumerate(documents):
    print(f"\nOriginal Document {i+1}: {doc}")
    print(f"Stemmed Document {i+1}: {stem_text(doc)}")
