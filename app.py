import os
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# Library yang digunakan:
# os        : mengecek keberadaan file model dan logo
# numpy     : mengolah array gambar
# streamlit : membuat tampilan aplikasi web
# tensorflow: memuat model CNN yang sudah dilatih
# PIL       : membaca dan mengubah ukuran gambar

# Mengatur tampilan awal halaman aplikasi
st.set_page_config(
    page_title="Aplikasi Klasifikasi Gambar CNN",
    page_icon="Logo Binus.jpg",
    layout="wide"
)

# Daftar kelas pada dataset CIFAR-10
CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# Nama file model hasil training
MODEL_PATH = "cnn_cifar10_optimized.keras"

# Bagian sidebar berisi informasi tugas dan identitas mahasiswa
with st.sidebar:
    if os.path.exists("Logo Binus.jpg"):
        st.image("Logo Binus.jpg", width=180)
    else:
        st.write("Binus University")

    st.markdown("### **Informasi Tugas**")
    st.write("**Week 8**")
    st.info("**TUGAS KELOMPOK 2: OPTIMASI DAN IMPLEMENTASI DALAM APLIKASI SEDERHANA**")
    st.markdown("---")
    st.write("**Nama:** TEJA LESMANA")
    st.write("**NIM:** 2902694521")

# Judul dan deskripsi singkat aplikasi
st.title("Aplikasi Klasifikasi Gambar CNN")
st.write("Program ini dibuat untuk mengoptimalkan model CNN dan mengintegrasikannya ke dalam antarmuka web sederhana.")

# Fungsi untuk memuat model
# Cache digunakan supaya model tidak dimuat ulang setiap kali halaman berubah.
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return tf.keras.models.load_model(MODEL_PATH)

# Memanggil model yang sudah disimpan dari proses training
model = load_model()

# Jika model belum tersedia, aplikasi menampilkan pesan peringatan
if model is None:
    st.warning(
        "File model belum ditemukan. Jalankan notebook training terlebih dahulu "
        "hingga menghasilkan file cnn_cifar10_optimized.keras, lalu simpan file tersebut "
        "di folder yang sama dengan app.py."
    )
else:
    # Informasi kelas yang dapat dikenali oleh model
    st.info("""
    **Catatan:** Model ini dilatih khusus untuk mengenali 10 kategori: 
    *Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck.*
    """)

    # Input gambar dari user
    uploaded_file = st.file_uploader(
        "Pilih gambar (Format: JPG, JPEG, PNG)",
        type=["jpg", "jpeg", "png"]
    )

    # Proses prediksi hanya dilakukan jika user sudah mengupload gambar
    if uploaded_file is not None:
        col1, col2 = st.columns(2)

        # Kolom kiri menampilkan gambar asli yang diupload
        with col1:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Gambar yang diupload", use_container_width=True)

        # Kolom kanan menampilkan hasil klasifikasi model
        with col2:
            # Gambar diubah ukurannya menjadi 32x32 pixel
            # karena model dilatih menggunakan dataset CIFAR-10
            img_resized = image.resize((32, 32))

            # Mengubah gambar menjadi array dan melakukan normalisasi piksel
            img_array = np.array(img_resized) / 255.0

            # Menambahkan dimensi batch agar sesuai dengan input model
            img_array = np.expand_dims(img_array, axis=0)

            # Melakukan prediksi menggunakan model CNN
            prediction = model.predict(img_array, verbose=0)[0]

            # Mengambil kelas dengan nilai probabilitas tertinggi
            predicted_index = int(np.argmax(prediction))
            predicted_class = CLASS_NAMES[predicted_index]
            confidence = float(prediction[predicted_index])

            # Menampilkan hasil prediksi utama
            st.subheader("Hasil Prediksi")
            st.success(f"Kelas prediksi: **{predicted_class}**")
            st.write(f"Confidence: **{confidence:.2%}**")

            # Menampilkan tiga kemungkinan kelas dengan probabilitas tertinggi
            st.subheader("Top 3 Prediksi")
            top3 = np.argsort(prediction)[-3:][::-1]
            for idx in top3:
                st.write(f"{CLASS_NAMES[idx]}: {prediction[idx]:.2%}")

# Bagian footer aplikasi
st.markdown("---")
st.caption("Tugas Kelompok 2 - Artificial Intelligence - Binus University Online 2026")
