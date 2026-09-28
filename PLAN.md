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

## Roadmap menuju Macro F1 0,60+

Target **0,60 atau lebih** adalah sasaran eksperimen, bukan hasil yang bisa dijamin. Posisi sekarang 0,53163 pada satu holdout lokal. Selisih ini cukup besar sehingga rencana berfokus pada kesalahan per kelas, sifat ordinal damage grade, dan validasi yang mengurangi overfit.

### Diagnosis awal

Pada eksperimen terbaik saat ini, F1 tiap kelas adalah: Grade 1 **0,659**, Grade 2 **0,365**, Grade 3 **0,415**, Grade 4 **0,495**, dan Grade 5 **0,724**. Prioritasnya memperbaiki Grade 2 dan 3 tanpa menurunkan Grade 1 dan 5. Kebingungan terbesar ada pada kelas berdekatan, terutama Grade 3↔4 dan Grade 4↔5; ini mendukung pengujian metode ordinal. Detail ada di confusion matrix `reports/validation_exp_002_weights.json`.

### Tahapan kerja dan gerbang skor

| Tahap | Perubahan yang diuji | Gerbang untuk lanjut |
|---|---|---|
| 1. Reproduksi dan iterasi CatBoost | Jalankan ulang konfigurasi depth 7 + `SqrtBalanced` memakai early stopping Macro F1 yang sudah ditambahkan. Uji 700–900 iterasi dengan learning rate 0,04–0,05 karena run sebelumnya menyentuh batas 500. | Skor lokal mengulang atau melewati 0,53163; bandingkan juga F1 tiap kelas. |
| 2. Bobot dan keputusan kelas | Bandingkan `SqrtBalanced`, `Balanced`, dan bobot manual. Simpan probabilitas; uji bias kelas, ambang, serta expected-grade decoding hanya pada data kalibrasi/OOF. | Peningkatan Macro F1 muncul pada prediksi OOF, bukan hanya pada satu validation set. |
| 3. Model ordinal | Karena urutan kelas bermakna, latih empat classifier kumulatif: `damage_grade > 1`, `> 2`, `> 3`, `> 4`. Ubah probabilitas kumulatif menjadi probabilitas 5 kelas, jaga urutannya monoton, lalu pilih batas keputusan dengan OOF. Bandingkan dengan CatBoost multiclass langsung. | Menaikkan terutama Grade 2/3 dan meningkatkan Macro F1 rata-rata OOF. |
| 4. Fitur terarah | Dari confusion matrix, uji fitur interaksi yang beralasan: umur × material, lantai × tinggi/luas, fondasi × kondisi tanah, serta konfigurasi struktur. Uji satu kelompok per run; fit statistik kategori hanya di fold train untuk mencegah leakage. | Peningkatan terulang pada beberapa fold/seed dan tidak mengorbankan kelas lain secara tajam. |
| 5. Ensemble beragam | Blend probabilitas OOF dari CatBoost multiclass, model ordinal, dan satu model tabular non-neural lain yang layak. Pilih bobot blend dengan OOF, jangan dengan leaderboard public. | Ensemble mengungguli model tunggal pada rerata Macro F1 OOF. |
| 6. Validasi akhir | Gunakan stratified 3-fold OOF untuk menyaring model; validasi 5-fold hanya untuk kandidat final bila waktu komputasi memungkinkan. Bandingkan rerata dan variasi skor. | Rerata OOF mendekati/menembus 0,60 dan skor tidak bergantung pada satu fold. |
| 7. Submission | Latih kandidat terpilih pada semua train, buat submission melalui notebook, verifikasi aturan `rules.md`, lalu unggah manual. | File benar; public score hanya pemeriksaan tambahan. Private leaderboard tetap penilaian akhir. |

Gunakan checkpoint sasaran **0,54 → 0,56 → 0,58 → 0,60**. Setelah setiap checkpoint, simpan report, parameter, seed, skor per kelas, confusion matrix, dan nama artefak. Jika suatu tahapan tidak meningkatkan OOF, jangan promosikan model; lanjut ke hipotesis berikutnya.

### Run pertama yang disarankan

Validasi dulu perubahan early stopping pada dua kandidat saja agar eksperimen tetap terukur:

```bash
PYTHONPATH=.packages python3 scripts/train.py \
  --experiment-id exp_003_iterations \
  --candidate '{"depth": 7, "learning_rate": 0.05, "iterations": 800, "l2_leaf_reg": 10, "auto_class_weights": "SqrtBalanced"}' \
  --candidate '{"depth": 7, "learning_rate": 0.04, "iterations": 900, "l2_leaf_reg": 12, "auto_class_weights": "SqrtBalanced"}'
```

Setelah itu, cek apakah `best_iteration` masih mencapai batas. Ulangi kandidat terbaik dengan seed berbeda, lalu implementasikan evaluasi probabilitas/OOF untuk langkah kalibrasi dan ordinal. Jangan langsung melatih model final atau mengganti submission hanya karena satu skor holdout naik.

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
