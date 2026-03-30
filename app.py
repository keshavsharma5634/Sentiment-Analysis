from flask import Flask, request, jsonify
import joblib
import numpy as np
from flask_cors import CORS
import re

app = Flask(__name__)
CORS(app)

model = joblib.load("model/sentiment_model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")


# 🔧 Clean text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# 🔧 Keywords
def extract_keywords(text):
    words = text.lower().split()
    return list(set([w for w in words if len(w) > 4]))[:3]


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    text = data.get("text")

    if not text:
        return jsonify({"error": "No text provided"}), 400

    # 🔥 CLEAN TEXT
    cleaned = clean_text(text)

    # 🔥 VECTORIZE
    vec = vectorizer.transform([cleaned])

    # 🔥 MODEL PREDICTION
    prediction = model.predict(vec)[0]
    probs = model.predict_proba(vec)[0]

    confidence = float(np.max(probs)) * 100

    # 🔥 KEYWORDS
    negative_keywords = ["scam", "fraud", "worst", "terrible", "bad", "refund", "fake"]
    positive_keywords = ["amazing", "best", "love", "great", "awesome",
    "excellent", "perfect", "satisfied", "happy",
    "wonderful", "fast", "good", "nice"
]

    # 🔥 FIXED LOGIC (IMPORTANT)
    if any(word in cleaned for word in negative_keywords):
        prediction = "negative"

    elif any(word in cleaned for word in positive_keywords):
        prediction = "positive"

    elif confidence < 45:
        prediction = "neutral"

    # 🔥 SCORE
    score = float(probs[1] - probs[0]) if len(probs) > 1 else 0

    return jsonify({
        "sentiment": prediction,
        "confidence": round(confidence, 2),
        "score": round(score, 2),
        "keywords": extract_keywords(text),
        "reason": "Predicted using Logistic Regression model"
    })


if __name__ == "__main__":
    app.run(debug=True)