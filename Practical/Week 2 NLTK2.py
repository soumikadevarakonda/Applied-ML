import string
import re
from nltk import word_tokenize
from nltk.corpus import stopwords
import nltk

nltk.download('punkt')
nltk.download('stopwords')

document = [
    "Hello, world! This is a test sentence. Let's see how it works.",
    "Another example sentence to demonstrate text processing.",
    "NLTK is a powerful library for natural language processing.",
    "Text preprocessing is essential for machine learning tasks."
]

stopwords = set(stopwords.words('english'))  
punctuation = set(string.punctuation)  

def clean_text(text):
    text = text.lower()  
    text = re.sub(r'[^\w\s]', '', text)  
    tokens = word_tokenize(text)  
    cleaned_tokens = [word for word in tokens if word not in stopwords and word not in punctuation]  
    return ' '.join(cleaned_tokens)  

cleaned_documents = [clean_text(doc) for doc in document] 
for i, doc in enumerate(cleaned_documents):
    print(f"Original Document {i+1}: {document[i]}")
    print(f"Cleaned Document {i+1}: {doc}\n")