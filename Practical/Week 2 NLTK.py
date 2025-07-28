import string
import re
from nltk import word_tokenize
from nltk.corpus import stopwords
import nltk

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

text = "Hello, world! This is a test sentence. Let's see how it works."

text = text.lower() 
text = re.sub(r'[^\w\s]', '', text) 
tokens = word_tokenize(text)  

stop_words = set(stopwords.words('english'))  
cleaned_tokens = [word for word in tokens if word not in stop_words] 

cleaned_text = ' '.join(cleaned_tokens) 
print("Original Text:", text)
print("Cleaned Text:", cleaned_text)