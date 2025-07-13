import pickle

model = pickle.load(open("models/spam_model.pkl", "rb"))
vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))

def predict_spam(text):
    vec = vectorizer.transform([text])
    result = model.predict(vec)
    return "Spam" if result[0] == 1 else "Not Spam"

msg = input("Enter an SMS: ")
print("Prediction:", predict_spam(msg))
