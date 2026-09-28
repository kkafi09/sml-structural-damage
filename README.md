# Structural Damage Classification

Proyek klasifikasi tingkat kerusakan bangunan pascagempa Nepal. Lihat [rules.md](rules.md) untuk spesifikasi tugas dan [PLAN.md](PLAN.md) untuk rencana iterasi.

## Data dan setup

Simpan file kompetisi (`train.csv`, `test.csv`, `sample_submission.csv`, `metadata.csv`) di `data/`. Data CSV diabaikan Git.

Gunakan Python 3.10 atau lebih baru dan instal dependensi dari `requirements.txt`. Lingkungan workspace saat ini menjalankan dependensi lokal dengan `PYTHONPATH=.packages`.

## Iterasi model

Setiap eksperimen diberi ID unik supaya laporan dan model tersimpan tanpa menimpa eksperimen lain. Contoh membandingkan tiga konfigurasi pada validation split stratified yang sama:

```bash
PYTHONPATH=.packages python3 scripts/train.py \
  --experiment-id exp_002_depth \
  --candidate '{"depth": 5, "learning_rate": 0.10, "iterations": 300, "l2_leaf_reg": 8}' \
  --candidate '{"depth": 6, "learning_rate": 0.08, "iterations": 400, "l2_leaf_reg": 8}' \
  --candidate '{"depth": 7, "learning_rate": 0.06, "iterations": 400, "l2_leaf_reg": 10}'
```

Laporan tersimpan sebagai `reports/validation_exp_002_depth.json`. Setelah memilih eksperimen yang ingin dijadikan kandidat submission, latih konfigurasi terbaik dari laporan itu pada seluruh data train:

```bash
PYTHONPATH=.packages python3 scripts/train.py --final --experiment-id exp_002_depth
```

Model versi eksperimen akan disimpan ke `models/exp_002_depth.joblib`. Jika Macro F1 validasinya mengungguli model lokal terbaik, skrip juga memperbarui `models/best_pipeline.joblib` dan `reports/best_model.json`. Untuk menguji class weighting, gunakan `--balanced-only --experiment-id exp_003_balanced`.

Jangan memakai ulang ID eksperimen kecuali memang ingin mengganti laporannya. Simpan konfigurasi terpilih, seed, dan skor untuk setiap iterasi.

## Submission Kaggle

Notebook `notebooks/03_submission.ipynb` menggunakan `EXPERIMENT_ID` untuk memuat model versi tertentu dan menyimpan file terpisah, misalnya `submissions/catboost_balanced_001.csv`. Periksa format, urutan `building_id`, dan label sebelum mengunggah file tersebut ke Kaggle. Catat skor public/private dan rank dari halaman Kaggle di `reports/submission_log.csv`.

Submission awal memperoleh **public score 0,52978** dan rank **6** pada screenshot leaderboard. Skor private belum diketahui. Public leaderboard memakai kira-kira 30% data test; hasil akhir ditentukan dari 70% sisanya, jadi gunakan validation Macro F1 lokal sebagai ukuran utama dan public score sebagai pemeriksaan sekunder.

## Model dan evaluasi

Workflow menggunakan CatBoost multiclass dengan feature engineering untuk umur, luas per lantai, dan tinggi per lantai. Kategori lokasi dan material diproses sebagai kategori native CatBoost. Model memakai objective multiclass; early stopping memakai `TotalF1:average=Macro` tanpa bobot agar selaras dengan metrik kompetisi. Kandidat dibandingkan dengan scikit-learn Macro F1 pada holdout stratified 20% dan seed 42. Validasi adalah estimasi, bukan jaminan skor private.

Hasil validasi awal:

| Konfigurasi | Macro F1 |
|---|---:|
| CatBoost depth 5, tanpa bobot | 0,52330 |
| CatBoost depth 6, tanpa bobot | 0,52397 |
| CatBoost depth 7, tanpa bobot | 0,52212 |
| CatBoost depth 6, `auto_class_weights=Balanced` | **0,52455** |

Model awal berbobot dilatih ulang pada seluruh 533.465 baris train.
