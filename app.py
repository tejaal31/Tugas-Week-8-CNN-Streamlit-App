""" import os
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

MODEL_PATH = "cnn_cifar10_optimized.keras"

st.set_page_config(page_title="Klasifikasi Gambar CIFAR-10", layout="centered")
st.title("Aplikasi Klasifikasi Gambar CNN")
st.write("Upload gambar, lalu model CNN akan memprediksi kelas gambar tersebut.")

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

if model is None:
    st.warning(
        "File model belum ditemukan. Jalankan notebook training terlebih dahulu "
        "hingga menghasilkan file cnn_cifar10_optimized.keras, lalu simpan file tersebut "
        "di folder yang sama dengan app.py."
    )
else:
    uploaded_file = st.file_uploader(
        "Pilih gambar", type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Gambar yang diupload", use_container_width=True)

        # Preprocessing: samakan ukuran gambar dengan input model CIFAR-10
        img_resized = image.resize((32, 32))
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array, verbose=0)[0]
        predicted_index = int(np.argmax(prediction))
        predicted_class = CLASS_NAMES[predicted_index]
        confidence = float(prediction[predicted_index])

        st.subheader("Hasil Prediksi")
        st.write(f"Kelas prediksi: **{predicted_class}**")
        st.write(f"Confidence: **{confidence:.2%}**")

        st.subheader("Top 3 Prediksi")
        top3 = np.argsort(prediction)[-3:][::-1]
        for idx in top3:
            st.write(f"{CLASS_NAMES[idx]}: {prediction[idx]:.2%}")
"""

import os
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
import pandas as pd

# Konfigurasi Halaman
st.set_page_config(
    page_title="CIFAR-10 Classifier",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS untuk mempercantik tampilan
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

MODEL_PATH = "cnn_cifar10_optimized.keras"

# Sidebar untuk Informasi Proyek
with st.sidebar:
    st.image("https://www.binus.ac.id/wp-content/uploads/2023/04/Logo-Binus-University.png", width=200)
    st.title("Tugas Kelompok 2")
    st.markdown("---")
    st.info("**Week 8: Optimasi & Implementasi CNN**")
    st.write("Aplikasi ini menggunakan model CNN yang telah dioptimasi dengan *Batch Normalization* dan *Dropout* untuk mengenali 10 jenis objek.")
    st.write("**Dataset:** CIFAR-10")

# Fungsi Load Model dengan Cache
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# Header Utama
st.title("🖼️ Klasifikasi Gambar Digital (CNN)")
st.write("Unggah gambar di bawah ini untuk melihat bagaimana kecerdasan buatan mengklasifikasikannya secara real-time.")

if model is None:
    st.error(f"❌ **File model '{MODEL_PATH}' tidak ditemukan!** Pastikan file sudah ada di repositori GitHub Anda.")
else:
    # Area Upload Gambar
    uploaded_file = st.file_uploader("Pilih gambar dari komputer Anda...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Layout Kolom
        col1, col2 = st.columns([1, 1], gap="large")

        with col1:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Gambar yang Diunggah", use_container_width=True)

        with col2:
            st.subheader("📊 Hasil Analisis Model")
            
            with st.spinner('Model sedang menganalisis pola gambar...'):
                # Preprocessing
                img_resized = image.resize((32, 32))
                img_array = np.array(img_resized) / 255.0
                img_array = np.expand_dims(img_array, axis=0)

                # Prediksi
                prediction = model.predict(img_array, verbose=0)[0]
                predicted_index = int(np.argmax(prediction))
                predicted_class = CLASS_NAMES[predicted_index]
                confidence = float(prediction[predicted_index])

                # Menampilkan Metrik Utama
                st.metric(label="Kelas Terdeteksi", value=predicted_class.upper())
                st.write(f"Tingkat Keyakinan: **{confidence:.2%}**")
                st.progress(confidence)

                # Visualisasi Top 3 Prediksi
                st.write("---")
                st.write("**Top 3 Probabilitas:**")
                top3_indices = np.argsort(prediction)[-3:][::-1]
                
                # Buat DataFrame untuk Bar Chart
                chart_data = pd.DataFrame({
                    'Label': [CLASS_NAMES[i] for i in top3_indices],
                    'Confidence': [float(prediction[i]) for i in top3_indices]
                }).set_index('Label')
                
                st.bar_chart(chart_data)

# Footer
st.markdown("---")
st.caption("Proyek Tugas Kelompok - Artificial Intelligence - 2026")
