from flask import Flask, render_template, request, redirect, url_for
import pickle
import pandas as pd
import csv
import os
from datetime import date

# Load trained model & vectorizer
model = pickle.load(open("models/spam_model.pkl", "rb"))
vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))

# Load dataset
df = pd.read_csv("data/spam.csv", encoding="ISO-8859-1")[["v1", "v2"]]
df.columns = ["label", "message"]

app = Flask(__name__)

# Track spam counts (per day basis)
spam_count = {}

@app.route("/", methods=["GET", "POST"])
def index():
    global spam_count
    prediction = None
    user_input = ""

    today = date.today().isoformat()
    if today not in spam_count:
        spam_count[today] = 0

    if request.method == "POST" and "message" in request.form:
        user_input = request.form["message"]
        if user_input.strip():
            vec_msg = vectorizer.transform([user_input])
            pred = model.predict(vec_msg)[0]
            if pred == 1:
                prediction = "🚨 Spam Message"
                spam_count[today] += 1
            else:
                prediction = "✅ Not Spam (Ham)"

    return render_template(
        "index.html",
        prediction=prediction,
        user_input=user_input,
        spam_count=spam_count[today]
    )

@app.route("/dataset")
def dataset():
    messages = df.copy()
    messages["prediction"] = model.predict(vectorizer.transform(messages["message"]))
    messages["prediction_label"] = messages["prediction"].map({0: "Ham ✅", 1: "Spam 🚨"})
    
    spam_total = (messages["prediction"] == 1).sum()
    ham_total = (messages["prediction"] == 0).sum()
    
    return render_template("dataset.html", 
                           tables=messages.to_dict(orient="records"),
                           spam_total=spam_total, 
                           ham_total=ham_total)

@app.route("/contact", methods=["POST"])
def contact():
    name = request.form["name"]
    email = request.form["email"]
    message = request.form["message"]

    # Save into CSV file
    os.makedirs("data", exist_ok=True)
    file_path = "data/contacts.csv"

    file_exists = os.path.isfile(file_path)
    with open(file_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Name", "Email", "Message"])  # header
        writer.writerow([name, email, message])

    return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(debug=True)
