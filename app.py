import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load('emotion_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Define bot responses
emotion_response = {
    'joy': "Yay! Keep spreading the joy! 🎉",
    'sadness': "I'm here for you. Things will get better soon. 💛",
    'anger': "Take a deep breath. Calmness will follow. 💨",
    'fear': "Don't worry. You are stronger than your fears. 💪",
    'love': "That's wonderful! Keep sharing the love! ❤️",
    'surprise': "Whoa! That sounds exciting! 🎉"
}

# --- Page Config ---
st.set_page_config(page_title="Emotion Detection Bot", layout="centered")

# --- Background Image Using CSS ---
page_bg_img = """
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("https://images.unsplash.com/photo-1503264116251-35a269479413?auto=format&fit=crop&w=1950&q=80");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
}

[data-testid="stHeader"] {
    background-color: rgba(0, 0, 0, 0);
}

.chatbot-title {
    font-family: 'Arial Black', sans-serif;
    color: #ffffff;
    font-size: 2.5em;
    text-align: center;
    padding: 0.5em;
}

.chatbox {
    background-color: rgba(0, 0, 0, 0.5);
    padding: 2em;
    border-radius: 15px;
    color: white;
}

.response {
    font-size: 1.2em;
    padding: 0.5em;
}
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)

# --- Dark/Light Theme Toggle ---
theme = st.toggle("🌙 Dark Theme / ☀️ Light Theme", value=True)

# Adjust text color based on theme
text_color = "#FFFFFF" if theme else "#000000"
bg_color = "rgba(0,0,0,0.6)" if theme else "rgba(255,255,255,0.6)"
font_color = "white" if theme else "black"

# Title
st.markdown(f'<h1 class="chatbot-title">Emotion Detection Chatbot 🤖</h1>', unsafe_allow_html=True)

# Chatbox container
st.markdown(f'<div class="chatbox">', unsafe_allow_html=True)

# --- Chat Interface ---
user_input = st.text_input("How are you feeling today?")

if user_input:
    cleaned = [' '.join(user_input.lower().split())]
    transformed_input = vectorizer.transform(cleaned)
    prediction = model.predict(transformed_input)[0]

    st.markdown(f'<div class="response"><b>Detected Emotion:</b> {prediction.title()}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="response">{emotion_response.get(prediction, "Stay positive! ✨")}</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
