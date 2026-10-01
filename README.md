# LnT Camp 2026 Final Project — Product Profitability Predictor

Proyek ini bertujuan membantu pengelola bisnis retail mengevaluasi profitabilitas produk secara lebih cepat dan terukur menggunakan Machine Learning, dibangun di atas dataset **Global Superstore**. Proyek ini dibuat sebagai bagian dari Final Project LnT Camp 2026, dengan tema *"Bridging the Gap: Empowering Future Talent through Machine Learning for Industry Innovation"*.

## Modelling Tasks

Proyek ini memilih **2 task** sesuai guideline:

1. **Regression** — Memprediksi rata-rata profit (`avg_profit`) suatu produk berdasarkan pola order historisnya (frekuensi order, kuantitas terjual, diskon, rata-rata penjualan, dan kategori produk).
2. **Classification** — Mengklasifikasikan produk ke dalam kategori **Profitable** atau **Not Profitable** berdasarkan pola order yang sama.

Kedua model menggunakan algoritma **Random Forest** dan dilatih pada level **granularitas produk** (bukan per transaksi), menggunakan data hasil agregasi dari 51.290 baris transaksi menjadi 10.292 produk unik.

## Struktur Folder

```
.
├── README.md
├── notebook/
│   └── tahap1_loadData.ipynb      # Notebook lengkap: EDA, preprocessing, modelling, evaluasi
├── model/
│   ├── model_reg.pkl               # Model Regression (Random Forest)
│   ├── model_clf.pkl               # Model Classification (Random Forest, class_weight='balanced')
│   └── feature_columns.pkl         # Daftar & urutan kolom fitur hasil One-Hot Encoding
├── backend/
│   ├── main.py                     # Backend API (FastAPI)
│   └── requirements.txt
└── frontend/
    ├── app.py                      # Frontend simulasi (Streamlit)
    └── requirements.txt
```

## Dataset

Dataset diambil dari [Global Superstore Dataset (Kaggle)](https://www.kaggle.com/datasets/fatihilhan/global-superstore-dataset), direstrukturisasi menjadi database SQLite ternormalisasi (`superstore.sqlite`, tidak disertakan di repo ini karena ukurannya besar — lihat `.gitignore`). File database dapat diunduh dari link yang disediakan oleh panitia LnT Camp 2026.

## Cara Menjalankan

### 1. Notebook

Buka `notebook/tahap1_loadData.ipynb`, letakkan file `superstore.sqlite` di folder yang sama, lalu jalankan seluruh cell secara berurutan dari atas ke bawah.

### 2. Backend API (FastAPI)

```bash
cd backend
pip install -r requirements.txt
python main.py
```

Backend akan berjalan di `http://localhost:8000`. Dokumentasi interaktif tersedia di `http://localhost:8000/docs`.

> **Catatan:** Backend tidak dideploy publik dan dijalankan secara lokal saat demo, sesuai ketentuan guideline *("The backend does not need to be publicly deployed — it may run locally during the demo")*.

### 3. Frontend (Streamlit)

```bash
cd frontend
pip install -r requirements.txt
streamlit run app.py
```

Buka `http://localhost:8501` di browser. Pastikan **backend sudah berjalan terlebih dahulu** di terminal terpisah sebelum membuka frontend.

**Frontend yang sudah dideploy (publik):** `<LINK_STREAMLIT_CLOUD_KAMU>`

> **Penting:** Karena backend dijalankan secara lokal (lihat catatan di atas), versi frontend yang dideploy di Streamlit Cloud **tidak dapat terhubung ke backend** (keterbatasan arsitektur: server Streamlit Cloud tidak dapat mengakses `localhost` di komputer lain). Link deployed di atas menunjukkan tampilan antarmuka aplikasi. **Untuk mencoba prediksi secara end-to-end**, silakan clone repository ini dan jalankan backend serta frontend secara bersamaan di komputer Anda sendiri, mengikuti langkah 2 dan 3 di atas.

## Ringkasan Hasil Model

| Model | Metrik Utama | Hasil |
|---|---|---|
| Regression | MAE / RMSE / R² | 47,34 / 133,68 / 31,98% |
| Classification | Accuracy / F1 / ROC-AUC | 79% / 0,86 (Profitable) / 83% |

Detail lengkap EDA, proses modelling, dan interpretasi bisnis tersedia di dalam notebook.

## Author

Dikerjakan secara mandiri sebagai bagian dari LnT Camp 2026 Final Project.
