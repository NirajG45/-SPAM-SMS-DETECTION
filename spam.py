from flask import Flask, render_template, request
import pickle

# Load model & vectorizer
model = pickle.load(open("models/spam_model.pkl", "rb"))
vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    user_input = ""
    if request.method == "POST":
        user_input = request.form["message"]
        if user_input.strip():
            # Transform input
            vec_msg = vectorizer.transform([user_input])
            pred = model.predict(vec_msg)[0]
            prediction = "🚨 Spam Message" if pred == 1 else "✅ Not Spam (Ham)"
    return render_template("index.html", prediction=prediction, user_input=user_input)

if __name__ == "__main__":
    app.run(debug=True)
