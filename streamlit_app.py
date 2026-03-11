import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load datasets
fake = pd.read_csv("dataset/Fake.csv")
true = pd.read_csv("dataset/True.csv")

# Label data
fake["label"] = 0
true["label"] = 1

# Combine datasets
data = pd.concat([fake, true])

X = data["text"]
y = data["label"]

# Convert text to numbers
vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(X)

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LogisticRegression()
model.fit(X_train, y_train)

# Streamlit UI
st.title("Fake News Detection App")

news_text = st.text_area("Enter news text")

if st.button("Check News"):
    input_data = vectorizer.transform([news_text])
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("This news is REAL")
    else:
        st.error("This news is FAKE")