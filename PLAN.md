# Rencana peningkatan Macro F1

## Status saat ini

- Tujuan eksperimen: Macro F1 lokal **lebih dari 0,60**. Tujuan belum tercapai dan iterasi model dijeda sementara untuk pengiriman submission.
- Skor holdout lokal terbaik: **0,598551** (holdout stratified 80/10/10, seed 2026), dari stack tiga XGBoost dan tiga LightGBM dengan logistic meta-model serta bias per kelas. Laporan stack: `reports/validation_exp_086_location_interaction_stack.json`; laporan kalibrasi: `reports/validation_exp_087_interaction_bias.json`.
- Kandidat submission saat ini: `submissions/best_stack_087.csv`. File berisi 228.629 prediksi sesuai urutan `building_id` di `sample_submission.csv`. Belum ada skor leaderboard Kaggle untuk kandidat ini.
- Skor Kaggle yang tercatat masih dari submission awal CatBoost: public **0,52978**, rank **6**; private score belum diketahui. Catatan ada di `reports/submission_log.csv`.
- Skor 0,598551 adalah skor validasi lokal, bukan jaminan skor Kaggle dan bukan skor yang diukur pada CSV submission. Refit submission menggunakan cache OOF untuk meta-model dan kalibrasi. Karena estimator LightGBM asli tidak disimpan, refit memakai jumlah iterasi yang dilaporkan dan persiapan fitur bersama sehingga tidak sepenuhnya identik dengan model validasi.

## Interpretasi hasil terbaik

F1 per kelas pada holdout eksperimen 086 adalah Grade 1 **0,7035**, Grade 2 **0,4730**, Grade 3 **0,4914**, Grade 4 **0,5649**, dan Grade 5 **0,7598**. Grade 2 dan 3 paling sulit diprediksi. Bias kelas yang dipilih hanya dari OOF calibration menaikkan skor evaluation holdout dari **0,590683** menjadi **0,598551**. Iterasi hingga eksperimen 115 belum melewati 0,60; eksperimen 115 memperoleh 0,597869.

Semua eksperimen memakai kelas damage grade 1–5 dan Macro F1 sebagai metrik utama. Pemilihan model, regularisasi, serta bias menggunakan data calibration/OOF; evaluation holdout dipakai untuk pengukuran akhir. Hindari mengoptimalkan langsung terhadap public leaderboard karena hanya sebagian test yang dinilai di sana.

## Rencana saat iterasi dilanjutkan

1. Reproduksi persis keenam model dasar dan simpan estimator serta konfigurasi, bukan hanya probabilitas holdout. Pastikan fitur dan urutan kolom train/test sama.
2. Evaluasi ulang kandidat terbaik pada beberapa seed holdout atau cross-validation OOF untuk mengukur kestabilan, terutama pada Grade 2 dan Grade 3.
3. Uji hipotesis fitur satu per eksperimen (struktur × umur, lantai × tinggi/luas, dan fondasi × lokasi) dengan seluruh encoding dipelajari hanya dari fold train.
4. Pertahankan batasan tugas: jangan gunakan neural network; gunakan Macro F1; gunakan `Pipeline` dan `ColumnTransformer` di workflow preprocessing/modeling; dokumentasikan setiap fungsi dan class Python.
5. Promosikan hasil baru hanya jika validasi yang sebanding menunjukkan peningkatan dan CSV submission memenuhi format di `rules.md`.

## Riwayat ringkas eksperimen

- CatBoost pipeline awal dan beberapa variasi mencapai sekitar 0,583 setelah kalibrasi, di bawah stack terbaik.
- XGBoost tunggal mencapai sekitar 0,5945. Target encoding lokasi–struktur dan stacking probabilitas meningkatkan hasil holdout ke 0,59854.
- Kalibrasi bias multi-start menaikkan holdout menjadi 0,598551. Slope kelas, decoding ordinal, CatBoost/ExtraTrees stack, fitur spasial tambahan, serta beberapa stack luas tidak memperbaiki skor tersebut.
- Uji lanjutan hingga `validation_exp_115_lgb_objectives_separate.json` menghasilkan 0,597869, sehingga kandidat exp087 tetap terbaik.

Detail metrik dan konfigurasi ada di `reports/validation_exp_*.json`. Cache `.npz` dipakai untuk membangun kembali stack dan tidak selalu memuat fitted estimator. Untuk mengirim submission, pertahankan dua kolom `building_id,damage_grade`, tanpa indeks tambahan, dan urutan `building_id` dari `sample_submission.csv`; catat hasil public/private score di `reports/submission_log.csv`.
