### Desrikpsi
Aplikasi ini dirancang untuk mengklasifikasikan varietas beras berdasarkan citra yang diunggah pengguna. Menggunakan teknologi deep learning dengan model InceptionV3 dan MobileNetV2, aplikasi dapat memprediksi varietas beras seperti Arborio, Basmati, Ipsala, Jasmine, dan Karacadag dengan tingkat kepercayaan yang ditampilkan dalam bentuk grafik. Tujuannya adalah memberikan alat bantu analisis cepat dan efisien dibidang pertanian untuk klasifikasi varietas beras.

### Instalasi
1. Clone repository ini:
    ```bash
    git clone https://github.com/FauzanHele/Klasifikasi-Varietas-Beras.git
    cd Klasifikasi-Varietas-Beras/src
    ```
2. Buat dan aktifkan virtual environment:
    ```bash
    python -m venv env
    source env/bin/activate  # Untuk Linux/Mac
    env\Scripts\activate    # Untuk Windows
    ```
3. Instal dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4. Jalankan aplikasi web:
    ```bash
    streamlit run app.py
    ```


### Model yang Digunakan
1. **InceptionV3**:
   - Dirancang untuk klasifikasi gambar dengan akurasi tinggi.
   - Ukuran input gambar: 224x224 piksel.
   - File model: `Model/Model_InceptionV3/model.h5`.

2. **MobileNetV2**:
   - Efisien untuk perangkat dengan sumber daya terbatas.
   - Ukuran input gambar: 224x224 piksel.
   - File model: `Model/Model_MobileNetV2/model.h5`.
## Analisis Performa
- Model telah diuji menggunakan dataset citra varietas beras.
- **InceptionV3** memberikan akurasi sebesar 99%, sedangkan **MobileNetV2** mencapai 99%.
- Evaluasi performa menggunakan metrik seperti akurasi, precision, recall, dan F1-score.


### Hasil dan Analisis
Berikut adalah hasil perbandingan model dalam bentuk grafik:
- **InceptionV3** menunjukkan performa lebih baik untuk klasifikasi kompleks.
![image](https://github.com/user-attachments/assets/48315657-e62e-4e06-9b3c-7f85d1478a07)

![image](https://github.com/user-attachments/assets/dd2b2fac-f817-4311-b8ff-5b5065406052)


- **MobileNetV2** unggul dalam kecepatan prediksi pada perangkat dengan sumber daya terbatas.
![image](https://github.com/user-attachments/assets/89e7d854-5f40-412e-aebd-6463e1c596a6)
  
![image](https://github.com/user-attachments/assets/b3ab1d18-282f-4c89-870d-c726d283bb05)



  
