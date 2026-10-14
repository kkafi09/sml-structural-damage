# Structural Damage Classification

Proyek ini memprediksi `damage_grade` (kelas 1–5) untuk bangunan setelah gempa Gorkha, Nepal. Metrik kompetisi adalah Macro F1. Rincian instruksi dan batasan tugas ada di [rules.md](rules.md).

## Pengumpulan 14 Oktober 2026

Model final pengumpulan adalah **stack eksperimen 087**, dengan Macro F1 validasi historis **0,598551**. Eksperimen lanjutan sampai 143 belum menghasilkan kandidat yang tervalidasi lebih baik. CSV final dipertahankan tanpa pelatihan ulang.

| Tujuan | Berkas yang dikumpulkan |
|---|---|
| Kaggle | `submissions/submission.csv` |
| GitHub | `README.md`, `.gitignore`, `requirements.txt`, `rules.md`, `PLAN.md`, tiga notebook, seluruh kode `scripts/`, laporan JSON, log submission, model pembanding, CSV prediksi |
| Lokal, diabaikan Git | Dataset `data/*.csv`, environment `.venv/` dan `.packages/`, kredensial, seluruh cache eksperimen `.npz`, snapshot CatBoost, checkpoint DART, dan log sementara |

Rujukan model final adalah [`reports/submission_best_stack_087.json`](reports/submission_best_stack_087.json). `models/best_pipeline.joblib`, `models/catboost_balanced_001.joblib`, dan `reports/best_model.json` merupakan artefak **baseline CatBoost** yang dipertahankan sebagai pembanding. Nama `best_pipeline` dan `best_model` berlaku pada workflow CatBoost tersebut; hasil final stack 087 tidak disimpan sebagai satu file estimator `.joblib`.

`PLAN.md` menjadi arsip eksperimen. Target Macro F1 0,62 belum tercapai. Kode, laporan JSON, notebook, dan CSV final disertakan untuk penilaian. Cache probabilitas `.npz` hanya disimpan lokal dan tidak disertakan dalam commit pengumpulan.

## Data dan instalasi

Unduh `train.csv`, `test.csv`, dan `sample_submission.csv` dari tab **Data** kompetisi Kaggle, lalu letakkan di `data/`. `ward_level_pred_intensity.csv` diperlukan untuk mereproduksi fitur intensitas dan lokasi; file ini tersedia di [repositori interpolasi intensitas gempa](https://github.com/P2i91/Earthquake-Damage-Grade-prediction_Kriging-Interpolation). Dataset tidak di-commit (`data/*.csv` diabaikan oleh Git). Akun Kaggle harus menyetujui aturan kompetisi sebelum mengunggah submission.

Gunakan Python 3.10 atau lebih baru:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
jupyter notebook
```

## Notebook dan skrip

Jalankan notebook sesuai urutan berikut:

1. [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb) memeriksa kolom, nilai hilang, distribusi target, dan ringkasan fitur.
2. [`notebooks/02_modeling.ipynb`](notebooks/02_modeling.ipynb) menjelaskan Pipeline/ColumnTransformer, membandingkan OvA dan OvO, dan menampilkan hasil model final yang dipilih.
3. [`notebooks/03_submission.ipynb`](notebooks/03_submission.ipynb) memvalidasi file CSV terbaik sebelum diunggah.

[`scripts/train.py`](scripts/train.py) memuat contoh workflow CatBoost dengan `Pipeline` dan `ColumnTransformer`, preprocessing di dalam pipeline, split stratified, dan evaluasi Macro F1. Skrip itu adalah workflow pembanding/pembelajaran; CSV terbaik berasal dari ensemble XGBoost–LightGBM dan meta-model OOF. Tidak ada neural network yang digunakan.

## Hasil model

Kandidat lokal terbaik saat ini adalah [`submissions/best_stack_087.csv`](submissions/best_stack_087.csv). Skor validasi historisnya **Macro F1 0,598551**, dari holdout stratified 80/10/10 dengan seed 2026. Pipeline akhir menggabungkan tiga model XGBoost dan tiga konfigurasi LightGBM melalui logistic meta-model; offset keputusan per kelas dipilih dari prediksi OOF calibration. Grade 2 dan 3 merupakan kelas tersulit. Ringkasan kandidat ada di [`reports/submission_best_stack_087.json`](reports/submission_best_stack_087.json), sedangkan skor per kelas dan confusion matrix tersedia pada laporan eksperimen lokal.

| Strategi | Macro F1 evaluation | Catatan |
|---|---:|---|
| LightGBM One-vs-All (OvA) | 0,588674 | Dengan kalibrasi kelas dari calibration split |
| Pairwise One-vs-One (OvO), Grade 2/3 dan 3/4 | 0,597164 | Koreksi pasangan kelas pada stack baseline |
| Ensemble OOF terpilih + kalibrasi | **0,598551** | Kandidat terbaik; nilai holdout lokal |

Angka di tabel adalah hasil konfigurasi terbaik yang tersedia untuk tiap strategi pada split seed 2026; perbandingan tidak mengisolasi strategi saja karena fitur dan model dasarnya berbeda. Data tabel dicatat di [`reports/model_strategy_comparison.json`](reports/model_strategy_comparison.json). Detail OvA/OvO serta interpretasi metrik dibahas di notebook modeling.

Skor lokal bukan skor Kaggle dan bukan jaminan skor test. Screenshot leaderboard 2 Oktober 2026 menunjukkan public score **0,59606** pada posisi ke-3 untuk kartu bernama `submission.csv`, tetapi nama itu belum dapat dipetakan dengan pasti ke CSV lokal. Private score pada screenshot tidak tersedia. Catatan submission ada di [`reports/submission_log.csv`](reports/submission_log.csv).

## Submission

Kandidat terbaik berversi adalah `submissions/best_stack_087.csv`. File siap unggah `submissions/submission.csv` sudah disamakan persis dengan kandidat tersebut. Keduanya memiliki kolom `building_id,damage_grade`, 228.629 baris, label 1–5, dan urutan ID dari `sample_submission.csv`. Jalankan notebook 03 untuk memeriksa kesamaan dan format, lalu unggah `submissions/submission.csv` secara manual melalui tab **Submissions** di Kaggle.

CSV final sudah tersedia sehingga penilaian format dan pengunggahan ke Kaggle tidak memerlukan cache atau pelatihan ulang. Untuk membangun ulang CSV, siapkan data sesuai petunjuk instalasi serta tujuh cache probabilitas lokal berikut di `reports/` (total sekitar 34,65 MB). Cache ini diabaikan Git dan tidak tersedia dalam clone baru:

- `stacking_holdout_086.npz`
- `xgboost_holdout_036.npz`, `xgboost_holdout_039.npz`, `xgboost_holdout_053.npz`
- `lightgbm_holdout_030.npz`, `lightgbm_holdout_041.npz`, `lightgbm_holdout_085.npz`

Jika cache belum tersedia, refit stack final belum dapat dijalankan; cache perlu disiapkan dari hasil eksperimen asli dengan konfigurasi dan split yang sama. Skrip eksperimen historis dapat sudah memakai konfigurasi eksperimen berikutnya, sehingga menjalankan versi terbarunya saja tidak menjamin regenerasi cache 030/036/039/041/053 yang identik.

Skrip juga menggunakan konfigurasi `reports/validation_exp_086_location_interaction_stack.json` dan offset kelas `reports/validation_exp_087_interaction_bias.json`. Setelah dependensi tersedia, jalankan:

```bash
python3 scripts/create_best_stack_submission.py
```

Skrip tersebut merefit model dasar dan menggunakan artefak OOF untuk meta-model serta kalibrasi. Karena estimator LightGBM asli tidak disimpan, refit memakai jumlah iterasi yang tercatat; prediksi hasil refit dapat berbeda dari model yang menghasilkan skor validasi. Skor 0,598551 adalah skor validasi resep eksperimen, bukan pengukuran ulang atas CSV ini. CSV yang ada tetap dapat diunggah tanpa menjalankan ulang refit.

Perintah refit menulis ulang `submissions/best_stack_087.csv` dan ringkasannya, tetapi tidak menyinkronkan `submissions/submission.csv` secara otomatis. Untuk pengumpulan ini gunakan CSV final yang sudah tersedia dan identik. Cache eksperimen lain diabaikan; menjalankan ulang eksperimen historis di luar jalur final memerlukan regenerasi cache terkait.

## Struktur repositori

- `data/`: data lokal kompetisi, tidak di-commit.
- `notebooks/`: EDA, pemodelan, dan pemeriksaan submission.
- `scripts/`: preprocessing, rekayasa fitur, validasi, stacking, dan pembuatan submission.
- `models/`: artefak baseline CatBoost dari workflow pipeline.
- `reports/`: laporan validasi, ringkasan kandidat final, log submission; cache `.npz` hanya lokal dan diabaikan Git.
- `submissions/`: CSV prediksi untuk Kaggle.
