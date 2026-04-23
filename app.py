import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os
import gdown

# =========================================
# 1. KONFIGURASI HALAMAN
# =========================================
st.set_page_config(page_title="Klasifikasi Sampah - Kamera", page_icon="♻️")

# =========================================
# 2. DOWNLOAD & LOAD MODEL (DRIVE)
# =========================================
MODEL_PATH = "model_skripsi_sampah.h5"

@st.cache_resource
def load_full_system():
    if not os.path.exists(MODEL_PATH):
        # GANTI ID DI BAWAH INI DENGAN ID FILE DRIVE ANDA
        file_id = 'MASUKKAN_ID_FILE_DRIVE_ANDA_DISINI'
        url = f'https://drive.google.com/file/d/18hc8WHannK0asctqfjgyFiDp2ZxbRsDY/view?usp=sharing'
        
        with st.spinner('Mendownload model dari Google Drive...'):
            gdown.download(url, MODEL_PATH, quiet=False)
    
    # Load model dengan compile=False agar lebih stabil
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

try:
    model = load_full_system()
except Exception as e:
    st.error(f"Gagal memuat sistem: {e}")
    model = None

# =========================================
# 3. INTERFACE UTAMA
# =========================================
st.title("♻️ Klasifikasi Sampah CNN")
st.write("Gunakan **Kamera** atau **Unggah File** untuk deteksi jenis sampah.")

# Pilihan input: Kamera atau File
option = st.radio("Pilih metode input:", ("Kamera (Live)", "Unggah Gambar"))

img_input = None

if option == "Kamera (Live)":
    img_input = st.camera_input("Ambil foto sampah")
else:
    img_input = st.file_uploader("Pilih file gambar...", type=["jpg", "jpeg", "png"])

# =========================================
# 4. PROSES PREDIKSI
# =========================================
if img_input is not None and model is not None:
    # Buka gambar
    img = Image.open(img_input)
    
    # Jika bukan dari kamera, tampilkan previewnya
    if option == "Unggah Gambar":
        st.image(img, caption='Gambar Berhasil Diunggah', use_container_width=True)

    # Tombol eksekusi
    if st.button('Mulai Deteksi'):
        with st.spinner('Menganalisis...'):
            # Preprocessing (Sesuai spek skripsi kamu 150x150)
            img_resized = img.resize((150, 150))
            img_array = image.img_to_array(img_resized)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0

            # Prediksi
            predictions = model.predict(img_array)
            score = predictions[0][0]

            st.write("---")
            # Mapping: 0 = Organik, 1 = Anorganik (Berdasarkan {'O': 0, 'R': 1})
            if score > 0.5:
                label = "ANORGANIK (RECYCLABLE)"
                confidence = score * 100
                st.error(f"### HASIL: {label}")
            else:
                label = "ORGANIK"
                confidence = (1 - score) * 100
                st.success(f"### HASIL: {label}")

            st.info(f"**Tingkat Keyakinan:** {confidence:.2f}%")

st.write("---")
st.caption("Aplikasi Klasifikasi Sampah - Proyek Skripsi")