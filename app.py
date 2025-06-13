
import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model dan tokenizer
model = tf.keras.models.load_model("emotion_model.h5")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Fungsi prediksi emosi
def predict_emotion(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=10, padding='post')
    pred = model.predict(padded)
    label = label_encoder.inverse_transform([np.argmax(pred)])
    return label[0], float(np.max(pred))

# UI Streamlit
st.set_page_config(page_title="Deteksi Emosi Teks", page_icon="😊", layout="centered")
st.title("🔍 Deteksi Emosi dari Teks")
st.write("Masukkan kalimat dan lihat apakah itu **positif**, **negatif**, atau **netral**!")

text_input = st.text_area("📝 Masukkan teks:", height=100)

if st.button("🔎 Deteksi"):
    if text_input.strip() == "":
        st.warning("Tolong masukkan teks terlebih dahulu.")
    else:
        label, confidence = predict_emotion(text_input)
        st.success(f"**Emosi Terdeteksi:** `{label.upper()}` dengan kepercayaan {confidence:.2f}")
        st.progress(int(confidence * 100))
