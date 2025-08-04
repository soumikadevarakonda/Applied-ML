import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('omw-1.4')

sentence = "The striped bats are hanging on their feet for best."

tokens = word_tokenize(sentence)

stemmer = PorterStemmer()
stemmed = [stemmer.stem(word) for word in tokens]

lemmatizer = WordNetLemmatizer()
lemmatized = [lemmatizer.lemmatize(word) for word in tokens]

pos_tags = pos_tag(tokens)

print("Original Tokens:\n", tokens)
print("Stemmed Tokens:\n", stemmed)
print("Lemmatized Tokens:\n", lemmatized)
print("POS Tags:\n", pos_tags)
