# Rencana iterasi model

## Status eksperimen

Eksperimen `exp_002_weights` membandingkan tiga konfigurasi pada holdout stratified 20%, seed 42. Skor terbaik sementara **0,53163** berasal dari CatBoost depth 7, learning rate 0,06, 500 iterasi, `l2_leaf_reg=10`, dan `auto_class_weights=SqrtBalanced`. Ini naik sekitar **0,00708** dari baseline validasi 0,52455. Ketiga kandidat mencapai batas jumlah iterasi, sehingga tahap berikutnya perlu menguji iterasi lebih banyak dengan learning rate lebih kecil. Hasil ini adalah skor validasi lokal; belum menjadi skor Kaggle dan belum mengubah model final maupun submission. Rincian ada di `reports/validation_exp_002_weights.json`.

## Baseline saat ini

- Submission awal: `catboost_balanced_001`.
- Skor public Kaggle dari screenshot: **0,52978**, posisi public leaderboard **6**.
- Macro F1 holdout lokal: **0,52455** (stratified 20%, seed 42).
- Model: CatBoost multiclass, depth 6, learning rate 0,10, 300 iterasi, `auto_class_weights=Balanced`.
- Artefak baseline disimpan sebagai `models/catboost_balanced_001.joblib` dan `submissions/catboost_balanced_001.csv`.
- Catatan hasil submission: `reports/submission_log.csv`.

Leaderboard pada screenshot menyatakan public score dihitung dari sekitar 30% data test dan hasil akhir memakai 70% sisanya. Karena itu public score dipakai sebagai pemeriksaan tambahan; jangan menyesuaikan model berulang kali hanya agar cocok dengan public leaderboard.

## Rencana peningkatan berikutnya

Lakukan tahapan berikut secara berurutan. Jalankan satu atau dua kandidat per eksperimen karena training pada seluruh data cukup lama. Gunakan ID baru dan simpan semua laporan.

1. **Pastikan early stopping sesuai metrik** — skrip sekarang memakai `TotalF1:average=Macro;use_weights=false`. Uji kembali konfigurasi terbaik `exp_002_weights` dengan skrip terbaru. Bandingkan Macro F1 hasil validasi scikit-learn, bukan hanya metrik CatBoost.
2. **Tambah iterasi secara terkendali** — kandidat terbaik mencapai batas 500 iterasi. Uji depth 7 dengan learning rate 0,04–0,06, `SqrtBalanced`, regularisasi 8–12, dan 700–900 iterasi. Catat `best_iteration`; jika masih selalu menyentuh batas, naikkan iterasi satu tahap lagi.
3. **Uji bobot kelas** — bandingkan `SqrtBalanced`, `Balanced`, serta bobot manual yang sedikit menguatkan kelas dengan recall terendah. Pilih bobot dari distribusi kelas train saja.
4. **Uji kestabilan** — setelah satu atau dua kandidat menang pada seed 42, jalankan ulang kandidat teratas pada seed 21 dan 2026. Gunakan rerata Macro F1 serta F1 tiap kelas sebelum menyebutnya peningkatan yang konsisten. Hindari memilih konfigurasi hanya dari satu split.
5. **Perbaiki keputusan kelas bila perlu** — simpan probabilitas validasi dan evaluasi penyesuaian ambang/perkalian probabilitas per kelas hanya pada data kalibrasi terpisah atau prediksi out-of-fold. Jangan menyetel ambang dan melaporkan skor pada baris validasi yang sama.
6. **Analisis kesalahan dan fitur** — gunakan confusion matrix untuk mencari pasangan kelas yang sering tertukar. Rancang fitur turunan yang beralasan dari luas, tinggi, umur, lantai, material, dan geografi; pertahankan fitur hanya jika peningkatan muncul di beberapa seed.
7. **Bandingkan strategi/model** — setelah baseline CatBoost stabil, bandingkan multiclass langsung dengan One-vs-Rest dan One-vs-One yang sesuai materi tugas, serta model boosting tabular lain yang tersedia. Semua metode harus dinilai pada split dan Macro F1 yang sama. Catat biaya komputasi karena OvO melatih lebih banyak model.
8. **Finalisasi** — latih konfigurasi yang menang pada seluruh data train, buat notebook submission dengan ID eksperimen yang sama, periksa format sesuai `rules.md`, dan baru unggah secara manual. Catat public score; jangan memilih hanya berdasarkan public leaderboard.

Contoh uji lanjutan pada kode terbaru:

```bash
PYTHONPATH=.packages python3 scripts/train.py \
  --experiment-id exp_003_iterations \
  --candidate '{"depth": 7, "learning_rate": 0.05, "iterations": 800, "l2_leaf_reg": 10, "auto_class_weights": "SqrtBalanced"}' \
  --candidate '{"depth": 7, "learning_rate": 0.04, "iterations": 900, "l2_leaf_reg": 12, "auto_class_weights": "SqrtBalanced"}'
```

Lalu bandingkan F1 laporan dengan baseline dan ulangi kandidat teratas pada seed berbeda sebelum training final.

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
- Skor terbaik mutlak tidak bisa dijamin. Pertahankan model lama sampai kandidat menang secara konsisten pada beberapa seed; skor validasi satu split dan public leaderboard bukan bukti final.

## Pemisahan workflow

- `scripts/train.py`: pencarian konfigurasi, validasi, training final, dan penyimpanan model versi eksperimen.
- `notebooks/03_submission.ipynb`: pemilihan versi model, prediksi test, pemeriksaan format, dan ekspor CSV.
- `reports/validation_<experiment_id>.json`: skor Macro F1 serta metrik per kelas untuk kandidat.
- `reports/submission_log.csv`: skor Kaggle dan hasil eksperimen yang dikirim.
- `models/` dan `submissions/`: artefak versi eksperimen; jangan menghapus baseline sebelum hasil final diumumkan.
