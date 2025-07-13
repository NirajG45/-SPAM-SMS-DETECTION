import pandas as pd
import string
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

df = pd.read_csv("data/spam.csv", encoding='ISO-8859-1')[["v1", "v2"]]
df.columns = ["label", "message"]
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

stop_words = set(stopwords.words('english'))

def clean_text(msg):
    msg = msg.lower()
    msg = "".join([ch for ch in msg if ch not in string.punctuation])
    msg = " ".join([word for word in msg.split() if word not in stop_words])
    return msg

df['clean_msg'] = df['message'].apply(clean_text)

X_train, X_test, y_train, y_test = train_test_split(df['clean_msg'], df['label'], test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train_vec, y_train)

preds = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, preds))
print(classification_report(y_test, preds))

pickle.dump(model, open("models/spam_model.pkl", "wb"))
pickle.dump(vectorizer, open("models/vectorizer.pkl", "wb"))
