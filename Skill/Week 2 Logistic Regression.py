import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils import resample

categories = ['rec.autos', 'sci.space', 'comp.graphics', 'talk.politics.misc']
data = fetch_20newsgroups(subset='all', categories=categories, remove=('headers', 'footers', 'quotes'))

X = data.data
y = data.target

df = pd.DataFrame({'text': X, 'target': y})
df_classe = df[df['target'] == 0]
df_others = df[df['target'] != 0]

df_class_downsampled = resample(df_classe, replace=False, n_samples=100, random_state=42)
df_balanced = pd.concat([df_class_downsampled, df_others])

X = df_balanced['text']
y = df_balanced['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.9)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model_lr = LogisticRegression(class_weight='balanced', max_iter=1000)
model_lr.fit(X_train_vec, y_train)
y_pred_lr = model_lr.predict(X_test_vec)
print("Logistic Regression Classification Report:\n")
print(classification_report(y_test, y_pred_lr, target_names=categories, zero_division=0))
disp_lr = ConfusionMatrixDisplay.from_estimator(model_lr, X_test_vec, y_test, display_labels=categories, cmap='Greens', xticks_rotation=45)
plt.title("Logistic Regression Confusion Matrix")
plt.show()