import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os

# Konfigurasi Halaman
st.set_page_config(page_title="Klasifikasi Sampah CNN", layout="centered")

st.title("🗑️ Klasifikasi Sampah Organik & Anorganik")
st.write("Unggah foto sampah untuk mendeteksi jenisnya.")

# Fungsi Load Model yang lebih aman
@st.cache_resource
def load_my_model():
    model_path = 'model_skripsi_sampah.h5'
    if os.path.exists(model_path):
        # compile=False menghindari error jika versi optimizer berbeda
        return tf.keras.models.load_model(model_path, compile=False)
    else:
        st.error("File model tidak ditemukan! Pastikan file .h5 ada di repositori GitHub.")
        return None

model = load_my_model()

# Upload Gambar
uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model is not None:
    # Menampilkan Gambar
    img = Image.open(uploaded_file)
    st.image(img, caption='Gambar Terpilih', use_container_width=True)
    
    st.write("---")
    
    with st.spinner('Sedang memproses...'):
        # Preprocessing
        img_resized = img.resize((150, 150))
        img_array = image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        # Prediksi
        predictions = model.predict(img_array)
        score = predictions[0][0]

        # Hasil (Berdasarkan Mapping {'O': 0, 'R': 1})
        if score > 0.5:
            label = "ANORGANIK (RECYCLABLE)"
            confidence = score * 100
            st.error(f"### Hasil: {label}")
        else:
            label = "ORGANIK"
            confidence = (1 - score) * 100
            st.success(f"### Hasil: {label}")

        st.write(f"**Tingkat Keyakinan:** {confidence:.2f}%")