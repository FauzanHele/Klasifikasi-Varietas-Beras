import numpy as np
import tensorflow as tf
from pathlib import Path
import streamlit as st
from io import BytesIO

# Konfigurasi halaman
st.set_page_config(
    page_title="Klasifikasi Citra Varietas Beras",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Tambahkan background gambar dengan custom CSS
st.markdown(
    """
    <style>
    body {
        background-image: url('src/assets/sawah.jpeg');
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    .stApp {
        background-color: rgba(255, 255, 255, 0.9);
        border-radius: 15px;
        padding: 15px;
    }
    h1 {
        color: #4CAF50;
        text-align: center;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.4);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Judul aplikasi
st.title("\U0001F33E Klasifikasi Citra Varietas Beras")
st.markdown(
    """<p style='text-align: center; color: #4CAF50; font-size: 20px;'>
    Unggah citra varietas beras untuk mendapatkan prediksi klasifikasi dan confidence level.
    </p>""",
    unsafe_allow_html=True,
)

# Fungsi prediksi
def predict(uploaded_image):
    # Daftar kelas
    class_names = ["Arborio", "Basmati", "Ipsala", "Jasmine", "Karacadag"]

    # Muat dan preprocess citra
    img = tf.keras.utils.load_img(uploaded_image, target_size=(224, 224))  # Pastikan ukuran sesuai dengan model
    img = tf.keras.utils.img_to_array(img) / 255.0  # Normalisasi
    img = np.expand_dims(img, axis=0)  # Tambahkan dimensi batch

    # Muat model
    model1_path = Path(__file__).parent / "Model\Model_InceptionV3\model.h5"
    model1 = tf.keras.models.load_model(model1_path)

    model2_path = Path(__file__).parent / "Model\Model_MobileNetV2\model.h5"
    model2 = tf.keras.models.load_model(model2_path)

    # Prediksi
    output = model1.predict(img)
    scores = tf.nn.softmax(output[0])  # Hitung probabilitas
    return class_names, scores.numpy()  # Daftar label dan confidence

# Fungsi untuk membuat grafik tanpa matplotlib
def plot_confidence_graph(class_names, scores):
    chart_data = {
        "Class": class_names,
        "Confidence": [score * 100 for score in scores],
    }
    st.bar_chart(chart_data, x="Class", y="Confidence", use_container_width=True)

# Komponen file uploader untuk banyak file
uploads = st.file_uploader(
    "\U0001F4C2 Unggah citra untuk mendapatkan hasil prediksi",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True,
)

# Tombol prediksi
if st.button("\U0001F4CA Prediksi", type="primary"):
    if uploads:
        st.subheader("\U0001F50D Hasil Prediksi")

        for upload in uploads:
            # Tata letak dua kolom
            col1, col2 = st.columns([1, 2])

            with col1:
                # Tampilkan citra yang diunggah
                st.image(upload, caption=f"Citra: {upload.name}", use_column_width=True)

            with col2:
                with st.spinner(f"Memproses citra {upload.name} untuk prediksi..."):
                    try:
                        # Panggil fungsi prediksi
                        class_names, scores = predict(upload)

                        # Hasil prediksi
                        predicted_class = class_names[np.argmax(scores)]
                        confidence = scores[np.argmax(scores)] * 100
                        st.success(f"\U0001F33E **Prediksi: {predicted_class}**")
                        st.info(f"Confidence: {confidence:.2f}%")

                        # Grafik
                        st.markdown("### Confidence Score")
                        plot_confidence_graph(class_names, scores)

                    except Exception as e:
                        st.error(f"\u274C Terjadi kesalahan saat memproses {upload.name}: {e}")
    else:
        st.error("\u26A0 Unggah setidaknya satu citra terlebih dahulu!")

# Sidebar informasi tambahan
st.sidebar.header("Tentang Aplikasi")
st.sidebar.markdown(
    """
    - **Teknologi**: TensorFlow, Streamlit
    - **Fitur**:
        - Klasifikasi varietas beras
        - Mendukung banyak citra
    - **Dikembangkan oleh**: [Fauzan](https://github.com)
    """
)

st.sidebar.header("Panduan")
st.sidebar.markdown(
    """
    1. Unggah citra varietas beras.
    2. Klik tombol **Prediksi**.
    3. Lihat hasil prediksi dan tingkat kepercayaan.
    """
)
