import pandas as pd
import string
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
import nltk
from nltk.corpus import stopwords
import os

nltk.download('stopwords')

# Load dataset
df = pd.read_csv("data/spam.csv", encoding='ISO-8859-1')[["v1", "v2"]]
df.columns = ["label", "message"]
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Stopwords
stop_words = set(stopwords.words('english'))

def clean_text(msg: str) -> str:
    """Clean input text: lowercase, remove punctuation, remove stopwords"""
    msg = msg.lower()
    msg = "".join([ch for ch in msg if ch not in string.punctuation])
    msg = " ".join([word for word in msg.split() if word not in stop_words])
    return msg

df['clean_msg'] = df['message'].apply(clean_text)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df['clean_msg'], df['label'], test_size=0.2, random_state=42
)

# Vectorization
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Model Training
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Evaluation
preds = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, preds))
print(classification_report(y_test, preds))

# Save model & vectorizer
os.makedirs("models", exist_ok=True)
pickle.dump(model, open("models/spam_model.pkl", "wb"))
pickle.dump(vectorizer, open("models/vectorizer.pkl", "wb"))

print("Model & Vectorizer saved successfully!")
