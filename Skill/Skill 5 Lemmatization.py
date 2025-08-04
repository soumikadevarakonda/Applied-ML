import nltk
from nltk.corpus import movie_reviews
import random
from nltk.classify import NaiveBayesClassifier
from nltk.classify.util import accuracy
from nltk.stem import PorterStemmer, WordNetLemmatizer

nltk.download('movie_reviews')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Load movie review documents and labels
docs = [(list(movie_reviews.words(fileid)), category)
        for category in movie_reviews.categories()
        for fileid in movie_reviews.fileids(category)]

random.shuffle(docs)

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def stem_features(words):
    return {stemmer.stem(w): True for w in words}

def lemma_features(words):
    return {lemmatizer.lemmatize(w): True for w in words}

stem_feats = [(stem_features(d), c) for (d, c) in docs]
lemma_feats = [(lemma_features(d), c) for (d, c) in docs]

train_stem, test_stem = stem_feats[:1500], stem_feats[1500:]
train_lemma, test_lemma = lemma_feats[:1500], lemma_feats[1500:]

stem_classifier = NaiveBayesClassifier.train(train_stem)
lemma_classifier = NaiveBayesClassifier.train(train_lemma)

print("\nAccuracy using Stemming:", accuracy(stem_classifier, test_stem))
print("Accuracy using Lemmatization:", accuracy(lemma_classifier, test_lemma))
