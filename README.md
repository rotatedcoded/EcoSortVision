# ♻️ EcoSort Vision

Aplikasi Computer Vision berbasis Streamlit untuk mengklasifikasikan sampah ke dalam 10 kategori menggunakan model **YOLO11 Classification**, guna membantu pengguna memilah sampah dengan lebih tepat dan mendukung praktik daur ulang.

## ✨ Fitur

- **Klasifikasi gambar sampah** ke dalam 10 kategori: baterai, sampah organik, kardus, pakaian, kaca, logam, kertas, plastik, sepatu, dan sampah campuran.
- **Dua mode input**: unggah gambar dari file, atau ambil foto langsung lewat kamera.
- **Panduan pengelolaan** untuk tiap kategori sampah yang terdeteksi.
- **Tingkat keyakinan (confidence)** ditampilkan beserta 3 prediksi teratas.
- Mendukung akselerasi **GPU (CUDA)** jika tersedia, otomatis fallback ke CPU.

## 📊 Performa Model

| Metrik | Nilai |
|---|---|
| Test accuracy | 94.24% |
| Jumlah kelas | 10 |
| Jumlah gambar test | 1.233 |

## 🗂️ Struktur Proyek

```
EcoSortVision/
├── app.py                  # Aplikasi utama Streamlit
├── models/
│   └── best.pt              # Model YOLO11 hasil training (di-ignore dari git)
├── data/
│   ├── raw/                 # Dataset mentah (di-ignore dari git)
│   └── prepared/             # Dataset yang sudah diproses (di-ignore dari git)
├── src/
│   ├── download_dataset.py   # Mengunduh dataset
│   ├── inspect_dataset.py    # Inspeksi/eksplorasi dataset
│   ├── prepare_dataset.py    # Praproses dataset
│   ├── train.py               # Melatih model
│   └── evaluate.py            # Evaluasi performa model
└── outputs/                  # Hasil training/evaluasi (di-ignore dari git)
```

## 🚀 Cara Menjalankan

1. **Clone repository**
   ```bash
   git clone https://github.com/rotatedcoded/EcoSortVision.git
   cd EcoSortVision
   ```

2. **Buat virtual environment & install dependencies**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate      # Windows
   pip install -r requirements.txt
   ```

3. **Jalankan aplikasi**
   ```bash
   streamlit run app.py
   ```

4. Buka browser ke `http://localhost:8501` dan mulai unggah gambar atau gunakan kamera.

## 🧠 Alur Kerja Model

Model klasifikasi dilatih menggunakan skrip di folder `src/`:
1. `download_dataset.py` — mengunduh dataset sampah.
2. `inspect_dataset.py` — mengecek distribusi dan kualitas data.
3. `prepare_dataset.py` — membagi dan memproses data untuk training.
4. `train.py` — melatih model YOLO11 Classification.
5. `evaluate.py` — mengevaluasi akurasi model pada data test.

Model hasil training disimpan di `models/best.pt` dan dimuat oleh `app.py` saat aplikasi dijalankan.

## 🛠️ Teknologi

- Streamlit — antarmuka web interaktif
- Ultralytics YOLO11 — model klasifikasi gambar
- PyTorch — backend deep learning
- Pandas & Pillow — pengolahan data dan gambar

## 📌 Catatan

Hasil prediksi merupakan bantuan klasifikasi awal. Selalu ikuti aturan pengelolaan sampah yang berlaku di daerah setempat.

## 👤 Kontributor

Dikembangkan oleh [@rotatedcoded](https://github.com/rotatedcoded)
