# ==========================================
# NLP Preprocessing & Text Classification
# Single File Implementation
# ==========================================

# -------- 1. Import Libraries --------
import pandas as pd
import numpy as np
import re
import nltk
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay
)

# Download NLTK data (run once)
nltk.download("stopwords")
nltk.download("punkt")
nltk.download("wordnet")

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# -------- 2. Load Dataset --------
# Example: SMS Spam Collection Dataset
# Make sure you have 'spam.csv' file in the same folder
df = pd.read_csv("spam.csv", encoding='latin-1')[['v1', 'v2']]
df.columns = ['label', 'text']

# Encode labels (spam=1, ham=0)
df['label'] = df['label'].map({'ham':0, 'spam':1})

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label'],
    test_size=0.2, random_state=42, stratify=df['label']
)

# -------- 3. Preprocessing Function --------
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()                             # lowercase
    text = re.sub(r"[^a-zA-Z]", " ", text)          # remove numbers/special chars
    tokens = nltk.word_tokenize(text)               # tokenize
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

# Apply preprocessing
X_train_clean = X_train.apply(preprocess_text)
X_test_clean = X_test.apply(preprocess_text)

# -------- 4. Feature Engineering --------
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train_clean)
X_test_vec = vectorizer.transform(X_test_clean)

# -------- 5. Model Training + Hyperparameter Tuning --------
param_grid = {"C": [0.1, 1, 10]}
grid = GridSearchCV(LogisticRegression(max_iter=1000), param_grid, cv=5, scoring="f1")
grid.fit(X_train_vec, y_train)

model = grid.best_estimator_

# -------- 6. Evaluation --------
y_pred = model.predict(X_test_vec)

print("\n--- Evaluation Metrics ---")
print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("F1-score :", f1_score(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Ham","Spam"])
disp.plot(cmap="Blues")
plt.show()

# -------- 7. Word Importance --------
feature_names = vectorizer.get_feature_names_out()
coefs = model.coef_[0]
top_features = np.argsort(coefs)[-10:]

print("\n--- Top Predictive Words for Spam ---")
for idx in top_features:
    print(feature_names[idx], ":", coefs[idx])
