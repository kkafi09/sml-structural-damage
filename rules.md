# Aturan Tugas Supervised Machine Learning

## Ringkasan

Tugas ini menggunakan kompetisi Kaggle untuk membangun model supervised learning dan mengirimkan prediksi. Hasil dinilai dari skor leaderboard serta kualitas dan kerapian repository GitHub.

## Soal dan tujuan

Kompetisi ini mengangkat penilaian kerusakan bangunan setelah gempa Gorkha, Nepal, pada 25 April 2015 (magnitudo 7,8 Mw). Data survei pascagempa dari Central Bureau of Statistics Nepal memuat karakteristik bangunan sebelum gempa, seperti struktur, bahan, umur, dan kondisi lokasi, beserta tingkat kerusakannya.

Tujuan tugas adalah membangun model klasifikasi supervised learning yang menggunakan fitur pada data bangunan untuk memprediksi `damage_grade` pada setiap `building_id` dalam `test.csv`. Kelas target yang harus diprediksi adalah **1, 2, 3, 4, atau 5**.

## Metrik evaluasi

Prediksi dievaluasi menggunakan **Macro F1-Score**. Metrik ini menghitung F1 untuk setiap kelas lalu merata-ratakannya, sehingga performa pada setiap tingkat kerusakan ikut diperhitungkan.

Saat validasi lokal, gunakan Macro F1 dan pastikan pembagian data mempertahankan proporsi kelas (misalnya `stratify=y`) jika sesuai. Semua preprocessing dan pemilihan model harus dipelajari hanya dari data training untuk mencegah kebocoran data.

## Format submission

File submission harus berformat CSV dengan header dan dua kolom berikut, dalam urutan yang sama:

```csv
building_id,damage_grade
122207000141,1
284909000871,2
241904000231,3
```

Buat satu prediksi `damage_grade` bernilai 1–5 untuk setiap `building_id` di `test.csv`. Pertahankan urutan dan nilai `building_id` dari `sample_submission.csv`; jangan sertakan indeks tambahan. Gunakan `sample_submission.csv` sebagai template kolom, lalu simpan dengan `index=False`.

## Kompetisi dan submission

- Gabung ke kompetisi melalui tautan yang dibagikan pengajar, lalu setujui aturan kompetisi di Kaggle.
- Unduh `train.csv`, `test.csv`, dan `sample_submission.csv` dari tab **Data**.
- Ikuti format kolom pada `sample_submission.csv` saat membuat `submission.csv`; simpan CSV tanpa indeks (`index=False`).
- Unggah submission melalui halaman kompetisi atau Kaggle CLI. Akun harus sudah menerima aturan kompetisi sebelum mengirim lewat API.
- Periksa hasil pada tab **Submissions** dan **Leaderboard**. Public score memakai sebagian data uji; private score memakai bagian lainnya dan menjadi penilaian setelah kompetisi berakhir.

## Aturan implementasi

- **Dilarang menggunakan model neural network.**
- Gunakan pipeline untuk preprocessing, training, dan tuning.
- Gunakan `Pipeline` dan `ColumnTransformer` dalam workflow.
- Setiap function atau class di file `.py` harus memiliki docstring.
- Jangan commit dataset besar (lebih dari 100 MB). Abaikan `data/*.csv` di `.gitignore` dan jelaskan cara memperoleh data di `README.md`.
- README harus menjelaskan cara menggunakan repository dan memuat interpretasi laporan/hasil.

## Struktur repository yang disarankan

```text
nama-kompetisi-kaggle/
├── README.md
├── .gitignore
├── requirements.txt
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_modeling.ipynb
│   └── 03_submission.ipynb
├── scripts/
│   ├── __init__.py
│   ├── preprocess.py
│   ├── feature_engineering.py
│   ├── train.py
│   └── predict.py
├── models/
├── submissions/
│   └── submission.csv
└── reports/
    └── figures/
```

Folder `data/` berisi data kompetisi, `notebooks/` berisi eksplorasi dan eksperimen, `scripts/` berisi kode modular, `models/` untuk model tersimpan, `submissions/` untuk file prediksi, dan `reports/figures/` untuk visualisasi hasil.

Contoh aturan `.gitignore` untuk data dan file sementara:

```gitignore
data/*.csv
!data/.gitkeep
__pycache__/
*.pyc
.ipynb_checkpoints/
venv/
.env
.DS_Store
```

## Materi multiclass

Dokumen menyertakan materi klasifikasi multi-kelas, termasuk strategi **One-vs-All (OvA)** dan **One-vs-One (OvO)**. Bandingkan strategi yang sesuai dengan data dan laporkan hasil evaluasinya. Materi ini merupakan referensi; larangan neural network tetap berlaku.

## Penilaian

| Komponen | Bobot |
|---|---:|
| Public score leaderboard | 15% |
| Private score leaderboard | 35% |
| Kesesuaian struktur, syntax, dan notebook di GitHub | 50% |

Bagian GitHub dinilai berdasarkan tree structure, workflow analisis, penggunaan Pipeline dan ColumnTransformer, docstring pada function/class, petunjuk penggunaan repository di README, serta laporan interpretasi di README.

## Referensi dalam dokumen tugas

- Tutorial Kaggle dan materi pendaftaran, verifikasi akun, bergabung ke kompetisi, mengunduh data, serta mengirim submission.
- Materi klasifikasi multi-kelas OvA dan OvO.
- Contoh repository: [github.com/m-tsaqif](https://github.com/m-tsaqif)
