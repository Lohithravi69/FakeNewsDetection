<<<<<<< HEAD
# train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score
import pickle
import os

# Load both datasets
fake_df = pd.read_csv(r"C:\Users\lohit\Documents\internship_VOC\week4\Fake.csv")
true_df = pd.read_csv(r"C:\Users\lohit\Documents\internship_VOC\week4\True.csv")

# Add label columns
fake_df['label'] = 'FAKE'
true_df['label'] = 'REAL'

# Combine datasets
df = pd.concat([fake_df, true_df], ignore_index=True)
print(df['label'].value_counts())

# Features and Labels
X = df['text']
y = df['label']

# Split BEFORE vectorization
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7, stratify=y)

# Vectorize only on training data
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Model
model = PassiveAggressiveClassifier(max_iter=50)
model.fit(X_train_vec, y_train)

# Create 'model' directory if it doesn't exist
os.makedirs("model", exist_ok=True)

# Save model and vectorizer
with open("model/fake_news_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("model/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

# Accuracy
y_pred = model.predict(X_test_vec)
print(f"Model Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print(df.head())
print(df['text'].isnull().sum())
=======
# train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score
import pickle
import os

# Load both datasets
fake_df = pd.read_csv(r"C:\Users\lohit\Documents\internship_VOC\week4\Fake.csv")
true_df = pd.read_csv(r"C:\Users\lohit\Documents\internship_VOC\week4\True.csv")

# Add label columns
fake_df['label'] = 'FAKE'
true_df['label'] = 'REAL'

# Combine datasets
df = pd.concat([fake_df, true_df], ignore_index=True)
print(df['label'].value_counts())

# Features and Labels
X = df['text']
y = df['label']

# Split BEFORE vectorization
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7, stratify=y)

# Vectorize only on training data
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Model
model = PassiveAggressiveClassifier(max_iter=50)
model.fit(X_train_vec, y_train)

# Create 'model' directory if it doesn't exist
os.makedirs("model", exist_ok=True)

# Save model and vectorizer
with open("model/fake_news_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("model/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

# Accuracy
y_pred = model.predict(X_test_vec)
print(f"Model Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print(df.head())
print(df['text'].isnull().sum())
>>>>>>> 20e4694479615c77a937492bd46ae1f76a9e2a53
