# Structural Damage Classification

Proyek ini memprediksi tingkat kerusakan bangunan setelah gempa Gorkha, Nepal. Target `damage_grade` memiliki lima kelas, dari 1 sampai 5. Metrik utama evaluasi adalah Macro F1. Rincian tugas dan batasan implementasi ada di [rules.md](rules.md).

## Data

Unduh `train.csv`, `test.csv`, `sample_submission.csv`, dan `metadata.csv` dari halaman **Data** kompetisi Kaggle, lalu simpan di `data/`. Eksperimen geospasial juga membutuhkan `ward_level_pred_intensity.csv`, yang tersedia dari [repositori interpolasi intensitas gempa](https://github.com/P2i91/Earthquake-Damage-Grade-prediction_Kriging-Interpolation). Dataset tidak disimpan di Git; pola `data/*.csv` sudah tercantum di `.gitignore`.

## Persiapan lingkungan

Gunakan Python 3.10 atau yang lebih baru, lalu pasang dependensi:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Pada workspace pengembangan ini, dependensi lokal dapat digunakan dengan `PYTHONPATH=.packages`.

## Menjalankan workflow

Workflow awal CatBoost tersedia di `scripts/train.py`. Skrip ini membangun `Pipeline` yang menggabungkan rekayasa fitur, `ColumnTransformer` untuk imputasi numerik dan pemrosesan kategorikal, serta classifier. Neural network tidak digunakan.

Contoh validasi beberapa kandidat dengan split stratified:

```bash
PYTHONPATH=.packages python3 scripts/train.py \
  --experiment-id exp_002_depth \
  --candidate '{"depth": 5, "learning_rate": 0.10, "iterations": 300, "l2_leaf_reg": 8}' \
  --candidate '{"depth": 6, "learning_rate": 0.08, "iterations": 400, "l2_leaf_reg": 8}' \
  --candidate '{"depth": 7, "learning_rate": 0.06, "iterations": 400, "l2_leaf_reg": 10}'
```

Laporan validasi disimpan di `reports/validation_<experiment-id>.json`. Setelah memilih kandidat yang tervalidasi, model dapat dilatih ulang dengan `--final` dan ID eksperimen yang sama. Skrip menyimpan model versi eksperimen di `models/` dan hanya memperbarui `models/best_pipeline.joblib` bila skor validasinya lebih baik.

Eksperimen stack terbaik direkam dalam `reports/validation_exp_086_location_interaction_stack.json`, lalu dikalibrasi menggunakan prediksi OOF calibration di `reports/validation_exp_087_interaction_bias.json`. Skrip `scripts/stack_location_interactions.py` dan `scripts/tune_interaction_stack_bias.py` mereproduksi tahapan tersebut dari cache probabilitas `.npz` yang dihasilkan eksperimen model dasar.

## Hasil dan interpretasi

Kandidat terbaik pada holdout lokal mencapai **Macro F1 0,598551** setelah bias per kelas. Holdout dibuat secara stratified 80/10/10 dengan seed 2026; regularisasi meta-model dipilih melalui 5-fold CV pada calibration set, dan bias kelas dipilih dari prediksi OOF calibration. Skor tersebut adalah estimasi lokal, bukan skor Kaggle.

F1 per kelas pada evaluasi holdout sebelum pencarian bias terakhir adalah: Grade 1 **0,7035**, Grade 2 **0,4730**, Grade 3 **0,4914**, Grade 4 **0,5649**, dan Grade 5 **0,7598**. Grade 2 dan 3 menjadi kelemahan terbesar. Pencarian bias multi-start menaikkan Macro F1 holdout dari **0,590683** menjadi **0,598551**; eksperimen setelahnya belum melampaui hasil ini. Laporan dan rincian tiap kelas disimpan di `reports/validation_exp_*.json`.

## Submission

Kandidat CSV yang tersedia adalah [`submissions/best_stack_087.csv`](submissions/best_stack_087.csv). CSV ini memiliki 228.629 baris, memakai kolom dan urutan ID dari `sample_submission.csv`, dan berisi prediksi integer 1–5. Ringkasannya ada di `reports/submission_best_stack_087.json`; skrip refit dan pembuatannya ada di `scripts/create_best_stack_submission.py`.

Untuk membangunnya kembali setelah menyiapkan semua data dan cache eksperimen:

```bash
PYTHONPATH=.packages python3 scripts/create_best_stack_submission.py
```

Periksa file CSV, lalu unggah melalui halaman kompetisi Kaggle setelah menyetujui aturan kompetisi. Notebook `notebooks/03_submission.ipynb` memvalidasi kandidat CSV yang sudah dibuat dan menampilkan ringkasan label sebelum submission diunggah.

**Catatan reproduksibilitas:** skor 0,598551 berasal dari evaluasi lokal eksperimen 087, bukan pengukuran terhadap CSV atau leaderboard Kaggle. Cache model dasar tersimpan sebagai probabilitas, bukan file fitted estimator. Karena itu XGBoost dan LightGBM direfit untuk menghasilkan prediksi test; khusus LightGBM, refit menggunakan jumlah iterasi tercatat dan persiapan fitur bersama sehingga dapat berbeda dari model persis yang menghasilkan skor holdout. Submission awal CatBoost memiliki public score **0,52978** dan rank **6**; kandidat `best_stack_087.csv` belum memiliki skor leaderboard yang tercatat. Catat hasil Kaggle berikutnya di `reports/submission_log.csv`.

## Struktur utama

- `data/`: data kompetisi lokal, tidak di-commit.
- `notebooks/`: eksplorasi, pemodelan, dan workflow submission.
- `scripts/`: rekayasa fitur, validasi, training, stacking, dan pembuatan submission.
- `models/`: model tersimpan dari workflow pipeline.
- `reports/`: metrik, confusion matrix, konfigurasi eksperimen, dan cache probabilitas.
- `submissions/`: file prediksi untuk diunggah ke Kaggle.
