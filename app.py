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

# Konfigurasi Halaman
st.set_page_config(
    page_title="Aplikasi Klasifikasi Gambar CNN",
    page_icon="Logo Binus.jpg",
    layout="wide"
)

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

MODEL_PATH = "cnn_cifar10_optimized.keras"

# Sidebar untuk Identitas Mahasiswa
with st.sidebar:
    if os.path.exists("Logo Binus.jpg"):
        st.image("Logo Binus.jpg", width=180)
    else:
        # Fallback jika nama file sedikit berbeda
        st.write("Binus University")
    st.markdown("### **Informasi Tugas**")
    st.write("**Week 8**")
    st.info("**TUGAS KELOMPOK 2: OPTIMASI DAN IMPLEMENTASI DALAM APLIKASI SEDERHANA**")
    st.markdown("---")
    st.write("**Nama:** TEJA LESMANA")
    st.write("**NIM:** 2902694521")

# Header Utama
st.title("Aplikasi Klasifikasi Gambar CNN")
st.write("Program ini dibuat untuk mengoptimalkan model CNN dan mengintegrasikannya ke dalam antarmuka web sederhana.")

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
    st.info("""
    **Catatan:** Model ini dilatih khusus untuk mengenali 10 kategori: 
    *Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck.*
    """)

    uploaded_file = st.file_uploader(
    "Pilih gambar (Format: JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        # Pembagian layout menjadi 2 kolom
        col1, col2 = st.columns(2)

        with col1:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Gambar yang diupload", use_container_width=True)

        with col2:
            # Preprocessing sesuai input model CIFAR-10[cite: 2]
            img_resized = image.resize((32, 32))
            img_array = np.array(img_resized) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Eksekusi Prediksi menggunakan logika asli Anda[cite: 2]
            prediction = model.predict(img_array, verbose=0)[0]
            predicted_index = int(np.argmax(prediction))
            predicted_class = CLASS_NAMES[predicted_index]
            confidence = float(prediction[predicted_index])

            st.subheader("Hasil Prediksi")
            st.success(f"Kelas prediksi: **{predicted_class}**")
            st.write(f"Confidence: **{confidence:.2%}**")

            st.subheader("Top 3 Prediksi")
            top3 = np.argsort(prediction)[-3:][::-1]
            for idx in top3:
                st.write(f"{CLASS_NAMES[idx]}: {prediction[idx]:.2%}")

# Footer Laporan
st.markdown("---")
st.caption("Tugas Kelompok 2 - Artificial Intelligence - Binus University Online 2026")
