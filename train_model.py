import pandas as pd
import joblib
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# 📥 Load dataset
df = pd.read_csv("data.csv", encoding="latin-1")

# Select columns
df = df[[0, 5]]
df.columns = ["sentiment", "text"]

# 🔥 Random 100k rows
df = df.sample(n=100000, random_state=42)


# 🧹 Clean text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()   # 🔥 extra cleaning
    return text


df["text"] = df["text"].apply(clean_text)

# 🎯 Map labels
df["sentiment"] = df["sentiment"].map({0: "negative", 4: "positive"})

# remove null
df = df.dropna()

# 📊 Check distribution
print("Class Distribution:\n", df["sentiment"].value_counts())


# ✂ Split data
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["sentiment"], test_size=0.2, random_state=42
)


# 🔠 TF-IDF
vectorizer = TfidfVectorizer(
    ngram_range=(1,2),
    max_features=20000,
    min_df=2,              # 🔥 improve
    stop_words="english"
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


# 🤖 Model
model = LogisticRegression(max_iter=500, class_weight="balanced")
model.fit(X_train_vec, y_train)


# 📈 Evaluation
y_pred = model.predict(X_test_vec)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))


# 💾 Save model
joblib.dump(model, "model/sentiment_model.pkl")
joblib.dump(vectorizer, "model/vectorizer.pkl")

print("\n🔥 Model trained & saved successfully!")