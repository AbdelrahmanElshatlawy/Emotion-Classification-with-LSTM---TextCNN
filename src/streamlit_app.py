import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
import os

# -----------------------------
# Load model & tokenizer
# -----------------------------
MODEL_PATH = r"E:\PROJECTS\NLP Project\models\emotion_lstm_model.h5"
TOKENIZER_PATH = r"E:\PROJECTS\NLP Project\tokenizer.pkl"

st.set_page_config(page_title="Emotion Classifier", layout="centered")

# Load model & tokenizer with clear errors (so hosting logs show what's wrong)
@st.cache_resource(show_spinner=False)
def load_model_and_tokenizer():
    # load model
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Place model under /models in the repo.")
    model = tf.keras.models.load_model(MODEL_PATH)

    # load tokenizer
    if not os.path.exists(TOKENIZER_PATH):
        raise FileNotFoundError(f"Tokenizer file not found at {TOKENIZER_PATH}. Place tokenizer.pkl in the repo root or update TOKENIZER_PATH.")
    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)

    return model, tokenizer


try:
    model, tokenizer = load_model_and_tokenizer()
except Exception as e:
    st.error(f"Failed to load model/tokenizer: {e}")
    st.stop()

class_names = ["Sadness", "Joy", "Love", "Anger", "Fear", "Surprise"]
max_length = 50


# -----------------------------
# Preprocessing function
# -----------------------------
def preprocess_text(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=max_length, padding='post')
    return padded


# -----------------------------
# Streamlit UI
# -----------------------------
st.title("Emotion Classification App")
st.write("Enter any text and the model will predict the emotion.")

# Text input box
user_input = st.text_input("Enter text:")

# Predict button
if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        processed = preprocess_text(user_input)
        prediction = model.predict(processed)
        predicted_class = class_names[np.argmax(prediction)]

        st.subheader("Predicted Emotion:")
        st.success(predicted_class)
