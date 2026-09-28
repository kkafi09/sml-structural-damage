# Rencana iterasi model

## Baseline saat ini

- Submission awal: `catboost_balanced_001`.
- Skor public Kaggle dari screenshot: **0,52978**, posisi public leaderboard **6**.
- Macro F1 holdout lokal: **0,52455** (stratified 20%, seed 42).
- Model: CatBoost multiclass, depth 6, learning rate 0,10, 300 iterasi, `auto_class_weights=Balanced`.
- Artefak baseline disimpan sebagai `models/catboost_balanced_001.joblib` dan `submissions/catboost_balanced_001.csv`.
- Catatan hasil submission: `reports/submission_log.csv`.

Leaderboard pada screenshot menyatakan public score dihitung dari sekitar 30% data test dan hasil akhir memakai 70% sisanya. Karena itu public score dipakai sebagai pemeriksaan tambahan; jangan menyesuaikan model berulang kali hanya agar cocok dengan public leaderboard.

## Siklus setiap iterasi

1. **Pilih satu hipotesis** — ubah satu kelompok hal saja, misalnya bobot kelas, kedalaman CatBoost, regularisasi, fitur turunan, atau jenis model tabular. Catat alasan sebelum run.
2. **Validasi dari `.py`** — gunakan split stratified. Early stopping CatBoost diarahkan ke Macro F1 tanpa bobot; kandidat kemudian dinilai dengan scikit-learn Macro F1. Simpan setiap run dengan ID unik beserta konfigurasi dan hasil validasinya.
3. **Periksa kelas dan kesalahan** — lihat `classification_report` dan confusion matrix; jangan mengejar kenaikan rata-rata jika satu kelas menjadi jauh lebih buruk.
4. **Latih kandidat final** — hanya finalisasi konfigurasi yang menunjukkan hasil lokal yang layak. Simpan model dengan ID eksperimen, bukan menimpa baseline.
5. **Buat submission dari notebook** — atur `EXPERIMENT_ID` agar sama dengan ID model, lalu jalankan `notebooks/03_submission.ipynb`. Notebook menghasilkan CSV dengan ID yang sama.
6. **Submit dan catat** — unggah CSV ke Kaggle, lalu tambahkan public score, private score bila sudah tersedia, posisi, tanggal, dan catatan ke `reports/submission_log.csv`.
7. **Promosikan kandidat terbaik** — pertahankan baseline jika kandidat tidak memberi perbaikan yang konsisten. `models/best_pipeline.joblib` hanya diperbarui oleh skrip saat Macro F1 validasi kandidat lebih tinggi daripada best lokal sebelumnya.

## Perintah untuk kandidat baru

Gunakan ID berbeda di setiap percobaan. Parameter kandidat dapat diberikan sebagai JSON berulang kali; contoh menguji dua konfigurasi depth dan regularisasi:

```bash
PYTHONPATH=.packages python3 scripts/train.py \
  --experiment-id exp_002_depth \
  --candidate '{"depth": 5, "learning_rate": 0.10, "iterations": 300, "l2_leaf_reg": 8}' \
  --candidate '{"depth": 7, "learning_rate": 0.06, "iterations": 400, "l2_leaf_reg": 10}'
```

Laporan akan disimpan sebagai `reports/validation_exp_002_depth.json`. Jika hasilnya menjanjikan, latih model pada seluruh data training dengan ID yang sama:

```bash
PYTHONPATH=.packages python3 scripts/train.py --final --experiment-id exp_002_depth
```

Lalu ubah `EXPERIMENT_ID` di `notebooks/03_submission.ipynb` menjadi `exp_002_depth` dan jalankan notebook. Hasilnya akan tersimpan sebagai `submissions/exp_002_depth.csv`, sehingga submission sebelumnya tetap tersedia.

Untuk menguji pembobotan kelas pada iterasi baru, jalankan `--balanced-only --experiment-id exp_003_balanced`; setelah itu gunakan `--final --experiment-id exp_003_balanced`.

## Aturan pemilihan dan penghentian

- Gunakan Macro F1 pada validasi lokal sebagai filter utama karena skor public hanya mewakili bagian test.
- Catat public score sebagai bukti tambahan, bukan sebagai satu-satunya dasar pemilihan. Private score menjadi penentu kompetisi ketika diumumkan.
- Jangan mengubah banyak hal sekaligus; setiap hasil harus bisa dihubungkan ke satu konfigurasi dan submission tertentu.
- Jangan menimpa model, submission, atau baris log eksperimen yang sudah ada.
- Skor terbaik mutlak tidak bisa dijamin dari satu split atau public leaderboard. Jika kandidat hanya unggul sangat tipis, uji ulang dengan seed lain sebelum menganggapnya peningkatan yang kuat.

## Pemisahan workflow

- `scripts/train.py`: pencarian konfigurasi, validasi, training final, dan penyimpanan model versi eksperimen.
- `notebooks/03_submission.ipynb`: pemilihan versi model, prediksi test, pemeriksaan format, dan ekspor CSV.
- `reports/validation_<experiment_id>.json`: skor Macro F1 serta metrik per kelas untuk kandidat.
- `reports/submission_log.csv`: skor Kaggle dan hasil eksperimen yang dikirim.
- `models/` dan `submissions/`: artefak versi eksperimen; jangan menghapus baseline sebelum hasil final diumumkan.
