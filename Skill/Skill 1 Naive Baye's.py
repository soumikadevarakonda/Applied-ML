import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_20newsgroups 
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.model_selection import train_test_split 
from sklearn.naive_bayes import ComplementNB 
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils import resample 

categories = ['rec.autos', 'sci.space', 'comp.graphics', 'talk.politics.misc'] 

data = fetch_20newsgroups(subset='all', categories=categories, 
remove=('headers', 'footers', 'quotes'))

X = data.data
Y = data.target

df = pd.DataFrame({'text': X, 'target': Y})
df_class0 = df[df['target'] == 0]
df_others = df[df['target'] != 0]
df_class0_downsampled = resample(df_class0, replace=False, n_samples=100, random_state=42) 
df_imbalanced = pd.concat([df_class0_downsampled, df_others])

X = df_imbalanced['text'] 
Y = df_imbalanced['target']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(stop_words='english', max_df=0.9)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = ComplementNB() 
model.fit(X_train_vec, Y_train) 
Y_pred = model.predict(X_test_vec) 

print("Classification Report:\n") 
print(classification_report(Y_test, Y_pred, target_names=categories, zero_division=0))


cm = confusion_matrix(Y_test, Y_pred) 
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=categories) 
disp.plot(xticks_rotation=45, cmap='Blues') 
plt.title("Confusion Matrix") 
plt.tight_layout() 
plt.show()