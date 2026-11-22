# Klasifikasi Emosi Ulasan Produk Tokopedia

**Proyek Akhir Praktikum Data Science 2026/2027**
Program Studi Informatika, UPN "Veteran" Yogyakarta | Kelompok 28

Metode: **Klasifikasi** (5 kelas emosi) | Fitur: **TF-IDF + Chi-Square** | Model: **Random Forest** dan **XGBoost**

---

## 1. Ringkasan Proyek

Ulasan produk Tokopedia memuat emosi pelanggan, tetapi jumlahnya terlalu banyak untuk dianalisis manual. Proyek ini membangun model klasifikasi yang membaca **teks ulasan** dan memprediksi salah satu dari lima emosi: **Happy, Love, Sadness, Fear, Anger**.

Poin utama proyek:

- Dataset: PRDECT-ID (5.400 ulasan, 11 kolom), dibersihkan menjadi 5.393 ulasan.
- Teks dinormalisasi secara konservatif (negasi, tanda baca, dan emoji tidak dihapus agar emosi tidak hilang), lalu diubah menjadi TF-IDF unigram + bigram dan disaring dengan Chi-Square (3.000 fitur).
- Data dibagi **berdasarkan `Product Name`** (`GroupShuffleSplit`) sehingga produk yang sama tidak muncul di data latih dan data uji.
- Hyperparameter dituning dengan `GroupKFold` (5 lipatan), dan model dipilih berdasarkan **Macro F1 pada validation set**. Data uji hanya dipakai satu kali untuk pelaporan akhir.
- Hasil: **Random Forest** terpilih, dengan Accuracy **62,3%** dan Macro F1 **0,596** pada data uji, jauh di atas baseline kelas mayoritas (Accuracy 33,6%, Macro F1 0,101).

## 2. Anggota dan Kontribusi

**Kelompok 28** ("abil, rapli, zaki, lintang tidur besok cari dataset lagi")

| No | Nama | NIM | Kontribusi |
|---|---|---|---|
| 1 | Muhammad Rafli Wibowo | 123240094 | Modelling, Hyperparameter Tuning, Deployment |
| 2 | Muhammad Aqbil Wildan Qinthara | 123240045 | EDA, Evaluasi, Penyusunan README |
| 3 | Muhammad Zaky Assyakuri | 123240222 | Business Understanding, Konklusi, Feature Selection |
| 4 | Lintang Cahaya Eka Putra | 123240245 | Data Gathering, Cleaning dan Preprocessing, Split Data |

## 3. Pertanyaan Penelitian

1. Bagaimana karakteristik dan distribusi lima kelas emosi pada ulasan produk Tokopedia dalam PRDECT-ID?
2. Bagaimana kinerja TF-IDF yang dikombinasikan dengan Random Forest dan XGBoost untuk mengklasifikasikan emosi ulasan?
3. Model mana yang memberikan Macro F1 terbaik pada validation set, dan bagaimana kinerjanya pada data uji?
4. Kelas emosi apa yang paling mudah dan paling sulit dikenali, serta pola kesalahan apa yang muncul?

Jawaban ada di bagian [Hasil dan Jawaban Pertanyaan Penelitian](#6-hasil-dan-jawaban-pertanyaan-penelitian).

## 4. Sumber Data

| Keterangan | Isi |
|---|---|
| Nama dataset | PRDECT-ID Dataset |
| Sumber | [Mendeley Data](https://data.mendeley.com/datasets/574v66hf2v/1) |
| Lisensi | CC BY 4.0 (ketentuan resmi mengikuti halaman sumber) |
| Ukuran | 5.400 baris, 11 kolom |
| Berkas di repository | `PRDECT-ID Dataset.csv` |

**Alasan pemilihan:** dataset menyediakan teks ulasan berbahasa Indonesia beserta label emosi yang dibutuhkan untuk *supervised text classification*, berasal dari sumber yang kredibel, dan bukan dataset praktikum, data sintetis, maupun data pribadi.

**Kondisi data ("kotor"):** 7 baris duplikat penuh, nilai ekstrem (outlier IQR) pada kolom numerik (Price 678 baris, Number Sold 532, Total Review 379, Overall Rating 252), dan 85 ulasan yang memuat karakter pengganti Unicode (U+FFFD) dari berkas sumber. Tidak ada missing value.

**Deskripsi kolom:**

| Kolom | Keterangan |
|---|---|
| `Category` | Kategori produk |
| `Product Name` | Nama produk, dipakai sebagai grup saat membagi data (bukan fitur) |
| `Location` | Lokasi yang tercatat pada data produk/ulasan |
| `Price` | Harga produk |
| `Overall Rating` | Rating keseluruhan produk |
| `Number Sold` | Jumlah produk terjual |
| `Total Review` | Jumlah ulasan produk |
| `Customer Rating` | Rating yang diberikan pelanggan |
| `Customer Review` | Teks ulasan pelanggan, **fitur utama model** |
| `Sentiment` | Label polaritas, **tidak dipakai** karena 100% ditentukan oleh `Emotion` (kebocoran target) |
| `Emotion` | **Target klasifikasi**: Happy, Love, Sadness, Fear, Anger |

## 5. Alur Analisis

Seluruh alur ada di notebook **`PROJECT_DS_UDAHFIX.ipynb`** dan mengikuti struktur wajib panduan proyek.

| Bagian panduan | Isi di notebook |
|---|---|
| 1. Business Understanding | Masalah, tujuan, pertanyaan penelitian, metode, metrik evaluasi (Macro F1, target ≥ 0,50), baseline |
| 2. Data Gathering | Load data, sumber dan lisensi, deskripsi kolom, alasan pemilihan dataset |
| 3. EDA | Statistik deskriptif, distribusi emosi, panjang ulasan, kata tersering per emosi, boxplot, korelasi, Cramér's V, representativitas sampel, beserta interpretasi di bawah tiap grafik |
| 4. Cleaning dan Preprocessing | Penanganan duplikat (before-after: 5.400 → 5.393), normalisasi teks, tabel keputusan seleksi fitur, split berbasis grup, encoding label, pembobotan kelas, TF-IDF + Chi-Square |
| 5. Modelling | Alasan pemilihan model, tuning `GroupKFold`, pemilihan model di validation set, training akhir, bukti reproducibility |
| 6. Evaluasi | Accuracy, Precision/Recall/F1 Macro, F1 Weighted, classification report, confusion matrix, tabel perbandingan akhir |
| 7. Konklusi | Jawaban pertanyaan penelitian, interpretasi metrik, keputusan, keterbatasan, saran |

### Keputusan penting

- **Fitur:** hanya teks ulasan (`clean_review`). `Sentiment` dibuang (kebocoran target). `Product Name` hanya untuk grup split. Kolom metadata lain tidak dipakai agar fokus pada klasifikasi teks.
- **Pembagian data:** `GroupShuffleSplit` 80:20 berdasarkan `Product Name` (4.399 latih, 994 uji), lalu data latih dibagi lagi menjadi subtraining 3.654 dan validation 745 (juga berbasis produk).
- **Tanpa kebocoran:** TF-IDF, Chi-Square, `LabelEncoder`, dan bobot kelas hanya di-*fit* pada data latih (pada tiap lipatan CV, hanya pada bagian latih lipatan).
- **Ketidakseimbangan kelas:** `class_weight="balanced"` (Random Forest) dan `sample_weight` (XGBoost), dengan Macro F1 sebagai metrik utama.
- **Jumlah fitur:** k = 3.000 (dipilih dari eksperimen pendahuluan pada validation set, lalu dikunci sebelum tuning dan perbandingan model).
- **Reproducibility:** satu `SEED = 42` untuk seluruh proses acak.

## 6. Hasil dan Jawaban Pertanyaan Penelitian

> Angka berikut berasal dari satu kali eksekusi notebook (Python 3.10.11, scikit-learn 1.7.2, xgboost 3.2.0). Hasil yang persis sama diharapkan pada versi library yang sama.

### Perbandingan model

| Model | Parameter terbaik | Macro F1 CV | Macro F1 validation | Accuracy uji | Macro F1 uji |
|---|---|---|---|---|---|
| Baseline (kelas mayoritas) | - | - | 0,092 | 0,336 | 0,101 |
| **Random Forest** | `n_estimators=400`, `max_features=log2`, `min_samples_leaf=3` | 0,553 | **0,579** | **0,623** | **0,596** |
| XGBoost | `max_depth=3`, `n_estimators=350` | 0,546 | 0,573 | 0,605 | 0,571 |

Dampak tuning (Macro F1 CV, parameter awal → terbaik): Random Forest 0,493 → 0,553 (naik 0,060), XGBoost 0,545 → 0,546 (hampir tidak berubah).

### Kinerja per kelas (Random Forest, data uji 994 ulasan)

| Emosi | Precision | Recall | F1 | Jumlah (support) |
|---|---|---|---|---|
| Happy | 0,774 | 0,740 | 0,757 | 334 |
| Sadness | 0,733 | 0,505 | 0,598 | 218 |
| Love | 0,553 | 0,631 | 0,589 | 157 |
| Anger | 0,481 | 0,602 | 0,535 | 128 |
| Fear | 0,462 | 0,548 | 0,502 | 157 |

Kesalahan terbanyak (label asli → prediksi): Happy → Love (65 ulasan), Sadness → Fear (45), Love → Happy (43).

### Jawaban pertanyaan penelitian

1. **Distribusi kelas:** tidak seimbang. Happy paling banyak (±33%), Anger paling sedikit (±13%). Karena itu dipakai pembobotan kelas dan Macro F1.
2. **Kinerja TF-IDF + RF/XGBoost:** Random Forest mencapai Accuracy 62,3% dan Macro F1 0,596, XGBoost 60,5% dan 0,571. Keduanya jauh di atas baseline, dan target Macro F1 ≥ 0,50 tercapai.
3. **Model terbaik:** Random Forest, dengan Macro F1 validation tertinggi (0,579 vs 0,573) dan juga unggul di CV dan data uji. Selisihnya kecil (0,025 pada data uji), sehingga tidak dapat dipastikan satu model lebih baik secara bermakna.
4. **Kelas mudah dan sulit:** Happy paling mudah dikenali (F1 0,76), Fear paling sulit (F1 0,50). Emosi negatif (Sadness, Fear, Anger) sering tertukar karena kosakatanya mirip.

### Arti hasil dalam bahasa awam

- **Accuracy 62,3%** berarti dari 100 ulasan baru, sekitar 62 ditebak benar emosinya. Menebak "Happy" untuk semua ulasan hanya benar ±34%.
- **Recall Sadness 50,5%** berarti sekitar setengah ulasan bernada sedih tidak berhasil dikenali.
- **Keputusan:** model cocok untuk memantau **tren emosi secara agregat**, tetapi belum cocok untuk mengambil keputusan otomatis per ulasan, terutama untuk emosi negatif.

## 7. Keterbatasan

- Hanya satu split uji (994 ulasan); belum ada validasi silang pada seluruh data.
- Masih ada teks ulasan identik lintas produk (dilaporkan, tidak dihapus otomatis).
- Label emosi dapat bersifat ambigu, sehingga ada batas atas akurasi yang realistis.
- TF-IDF tidak menangkap konteks kalimat, sarkasme, dan ejaan informal.
- Dataset terkonsentrasi pada penjual di wilayah Jabodetabek, sehingga belum mewakili seluruh pengguna Tokopedia.
- Jumlah fitur (k) dipilih dari grid terbatas dan hyperparameter dituning pada grid kecil.

## 8. Struktur Repository

```
.
├── README.md
├── requirements.txt
├── PROJECT_DS_UDAHFIX.ipynb      # notebook lengkap (Business Understanding sampai Konklusi)
└── PRDECT-ID Dataset.csv         # dataset
```

Saat notebook dijalankan, bagian Deployment membuat folder `deployment_artifacts/` berisi model terlatih (`.joblib`). Folder ini dihasilkan otomatis dan tidak wajib disertakan di repository.

## 9. Cara Menjalankan

1. **Clone repository**
   ```bash
   git clone <url-repository-anda>
   cd <nama-folder-repository>
   ```

2. **(Opsional) Buat virtual environment**
   ```bash
   python -m venv venv
   # Windows: venv\Scripts\activate
   # Linux/Mac: source venv/bin/activate
   ```

3. **Install dependensi**
   ```bash
   pip install -r requirements.txt
   ```

4. **Buka notebook dan jalankan seluruh sel secara berurutan**
   ```bash
   jupyter notebook PROJECT_DS_UDAHFIX.ipynb
   ```
   Pilih *Kernel → Restart & Run All*. Pastikan `PRDECT-ID Dataset.csv` berada di folder yang sama dengan notebook.

**Catatan waktu:** tahap tuning hyperparameter (GroupKFold) memakan waktu beberapa menit (±9 menit pada eksekusi kami), tergantung spesifikasi komputer.

## 10. Referensi

- Dataset: PRDECT-ID, [Mendeley Data](https://data.mendeley.com/datasets/574v66hf2v/1), lisensi CC BY 4.0.
- Panduan: Panduan Proyek Akhir Praktikum Data Science 2026/2027, Prodi Informatika UPN "Veteran" Yogyakarta.
