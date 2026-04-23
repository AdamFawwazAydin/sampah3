import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import InputLayer
import numpy as np
from PIL import Image
import os
import gdown
import cv2

# =========================================
# 1. KONFIGURASI HALAMAN
# =========================================
st.set_page_config(page_title="Klasifikasi Sampah - Fixed", page_icon="♻️")

# =========================================
# 2. DOWNLOAD & LOAD MODEL (FIXED)
# =========================================
MODEL_PATH = "model_skripsi_sampah.h5"

@st.cache_resource
def load_full_system():
    # Download dari Drive jika file tidak ada
    if not os.path.exists(MODEL_PATH):
        # GANTI ID DI BAWAH INI DENGAN ID FILE DRIVE KAMU
        file_id = '1fs5cqFvZyXorbs6fZaxWvGOrKtFfodlb' 
        url = f'https://drive.google.com/file/d/18hc8WHannK0asctqfjgyFiDp2ZxbRsDY/view?usp=drive_link'
        with st.spinner('Mendownload model dari Google Drive...'):
            gdown.download(url, MODEL_PATH, quiet=False)
    
    # Fungsi ini memaksa Keras mengabaikan parameter 'batch_shape' & 'optional' yang bikin error
    def fix_input_layer(config):
        config.pop('batch_shape', None)
        config.pop('optional', None)
        return InputLayer.from_config(config)

    # Memuat model dengan Custom Object Scope
    try:
        with tf.keras.utils.custom_object_scope({'InputLayer': fix_input_layer}):
            model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        return model
    except Exception as e:
        # Jika masih gagal, gunakan cara alternatif
        return tf.keras.models.load_model(MODEL_PATH, compile=False)

try:
    model = load_full_system()
except Exception as e:
    st.error(f"Sistem gagal dimuat: {e}")
    model = None

# =========================================
# 3. INTERFACE UTAMA
# =========================================
st.title("♻️ Klasifikasi Sampah CNN")
st.write("Gunakan kamera atau unggah gambar untuk deteksi.")

option = st.selectbox("Pilih Metode Input:", ("Kamera (Live)", "Unggah Gambar"))

img_input = None

if option == "Kamera (Live)":
    img_input = st.camera_input("Ambil foto sampah")
else:
    img_input = st.file_uploader("Pilih file gambar...", type=["jpg", "jpeg", "png"])

# =========================================
# 4. PREDIKSI
# =========================================
if img_input is not None and model is not None:
    img = Image.open(img_input)
    
    if option == "Unggah Gambar":
        st.image(img, caption='Gambar Terpilih', use_container_width=True)

    if st.button('Mulai Deteksi'):
        with st.spinner('Menganalisis...'):
            # Pastikan format ke RGB
            img_array = np.array(img.convert('RGB'))
            # Resize ke 150x150 sesuai training
            img_resized = cv2.resize(img_array, (150, 150))
            # Normalisasi
            img_final = img_resized / 255.0
            img_final = np.expand_dims(img_final, axis=0)

            # Prediksi
            predictions = model.predict(img_final)
            score = predictions[0][0]

            st.write("---")
            if score > 0.5:
                st.error(f"### HASIL: ANORGANIK")
                st.write(f"Tingkat Keyakinan: {score * 100:.2f}%")
            else:
                st.success(f"### HASIL: ORGANIK")
                st.write(f"Tingkat Keyakinan: {(1 - score) * 100:.2f}%")

st.write("---")
st.caption("Aplikasi Skripsi - Klasifikasi Sampah Adam Fawwaz Aydin")