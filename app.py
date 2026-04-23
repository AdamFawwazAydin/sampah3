import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# =========================================
# KONFIGURASI HALAMAN
# =========================================
st.set_page_config(page_title="Klasifikasi Sampah CNN", layout="centered")

st.title("🗑️ Klasifikasi Sampah Organik & Anorganik")
st.write("Unggah foto sampah untuk mengetahui kategorinya.")

# =========================================
# LOAD MODEL
# =========================================
@st.cache_resource
def load_my_model():
    # Pastikan nama file model sesuai dengan yang kamu simpan
    return tf.keras.models.load_model('model_skripsi_sampah.keras')

model = load_my_model()

# =========================================
# UPLOAD GAMBAR
# =========================================
uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Menampilkan gambar yang diunggah
    img = Image.open(uploaded_file)
    st.image(img, caption='Gambar yang diunggah', use_column_width=True)
    
    st.write("---")
    st.write("🔄 Sedang mengklasifikasi...")

    # Preprocessing Gambar
    # Sesuaikan target_size dengan yang ada di modelmu (150x150)
    img_resized = img.resize((150, 150))
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalisasi sama dengan saat training

    # Prediksi
    prediction = model.predict(img_array)
    
    # Menentukan Hasil (Asumsi: 0 = Organik, 1 = Anorganik/Recyclable)
    # Cek Mapping kelasmu sebelumnya di Colab {'O': 0, 'R': 1}
    if prediction[0][0] > 0.5:
        result = "ANORGANIK / RECYCLABLE"
        confidence = prediction[0][0] * 100
        st.error(f"Hasil: {result}")
    else:
        result = "ORGANIK"
        confidence = (1 - prediction[0][0]) * 100
        st.success(f"Hasil: {result}")

    st.write(f"Tingkat Keyakinan: {confidence:.2f}%")