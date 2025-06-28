import streamlit as st
import joblib

# Load trained model and vectorizer
model = joblib.load("hate_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# App title
st.title("🚨 Hate Speech Detector")

# Text input field
text = st.text_area("Enter your text:")

# Predict button
if st.button("Detect"):
    # Vectorize input text
    vec = vectorizer.transform([text])
    # Get prediction probability for class 0 (hate speech)
    proba = model.predict_proba(vec)[0][0]

    # Custom threshold: 0.4 for stricter detection
    if proba > 0.4:
        st.error(f"⚠️ Hate Speech Detected! (Confidence: {proba:.2f})")
    else:
        st.success(f"✅ Clean Content (Confidence: {proba:.2f})")
