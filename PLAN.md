# Rencana perbaikan Macro F1

## Posisi saat ini

- Submission Kaggle saat ini: **Macro F1 0,52978**, peringkat public 6 (sesuai screenshot).
- Validasi holdout stratified 20%, seed 42: **0,53163**, konfigurasi CatBoost depth 7, learning rate 0,06, 500 iterasi, `l2_leaf_reg=10`, `SqrtBalanced`.
- Kandidat tersebut mencapai batas 500 iterasi, jadi belum diketahui apakah performa sudah mencapai titik terbaik.
- F1 validasi terendah ada di Grade 2 (**0,365**) dan Grade 3 (**0,415**). Grade 1 (**0,659**) dan Grade 5 (**0,724**) jauh lebih baik.
- Skor publik hanya dipakai sebagai sinyal tambahan. Pemilihan model utama harus didasarkan pada validasi lokal yang stabil, sebab private leaderboard memakai data test yang berbeda.

## Sasaran

Naikkan Macro F1 secara bertahap: **0,54 → 0,56 → 0,58**. Angka tersebut target eksperimen, bukan jaminan. Jangan mengirim model baru hanya karena satu split atau satu skor public naik.

## Rencana kerja berurutan

### 1. Perpanjang pencarian CatBoost

Uji dua konfigurasi yang melanjutkan kandidat terbaik dengan learning rate lebih rendah dan iterasi lebih banyak:

```bash
PYTHONPATH=.packages python3 scripts/train.py \
  --experiment-id exp_003_iterations \
  --candidate '{"depth": 7, "learning_rate": 0.05, "iterations": 800, "l2_leaf_reg": 10, "auto_class_weights": "SqrtBalanced"}' \
  --candidate '{"depth": 7, "learning_rate": 0.04, "iterations": 900, "l2_leaf_reg": 12, "auto_class_weights": "SqrtBalanced"}'
```

**Lulus jika:** Macro F1 validasi setidaknya menyamai 0,53163, F1 per kelas tidak memburuk tajam, dan iterasi terbaik tidak terus menempel pada batas. Jika masih mencapai batas, naikkan iterasi sekali lagi; jika tidak, simpan konfigurasi terbaik sebagai pembanding.

### 2. Ukur kestabilan lintas pembagian data

Jalankan konfigurasi terbaik dari tahap 1 pada beberapa seed holdout, lalu gunakan Stratified 3-fold OOF untuk kandidat yang menjanjikan. Catat Macro F1 per fold, rata-rata, variasi skor, dan F1 per kelas. Gunakan split/fold yang sama untuk membandingkan semua metode.

**Lulus jika:** kenaikan rata-rata OOF konsisten dan tidak datang dari satu fold saja. Jangan memilih kandidat dengan skor puncak tinggi tetapi hasil fold lain jatuh.

### 3. Kalibrasi prediksi untuk Macro F1

Simpan probabilitas OOF model, lalu bandingkan `argmax` dengan keputusan yang dikalibrasi: bias per kelas atau penyesuaian ambang. Cari parameter hanya pada prediksi OOF; jangan gunakan label validasi yang sama untuk melatih model dan memilih threshold.

**Lulus jika:** Macro F1 OOF meningkat dan peningkatan tetap tampak di fold evaluasi terpisah. Simpan parameter kalibrasi bersama model.

### 4. Uji pendekatan ordinal

Karena Grade 1–5 memiliki urutan, latih empat classifier kumulatif untuk `damage_grade > 1`, `> 2`, `> 3`, dan `> 4`. Ubah probabilitas kumulatif menjadi probabilitas tiap grade, terapkan konsistensi urutan, lalu tentukan grade dengan aturan yang dipilih dari OOF. Bandingkan dengan CatBoost multiclass pada fold yang sama.

**Lulus jika:** rata-rata Macro F1 OOF lebih tinggi, terutama dengan perbaikan Grade 2/3 dan tanpa penurunan besar Grade 1/5.

### 5. Tambahkan fitur berbasis hipotesis

Uji satu kelompok fitur per eksperimen agar dampaknya dapat ditelusuri:

- umur × material/superstructure;
- jumlah lantai × tinggi dan luas bangunan;
- fondasi × kondisi permukaan/lokasi;
- ringkasan konfigurasi struktur yang sudah ada.

Fit seluruh statistik transformasi hanya pada bagian train di tiap fold untuk mencegah kebocoran. Jangan memasukkan `building_id` sebagai prediktor tanpa bukti bahwa pola ID akan berlaku di test.

**Lulus jika:** peningkatan muncul di lebih dari satu fold dan dapat dijelaskan melalui metrik per kelas.

### 6. Uji model dan ensemble

Sebagai pembanding, uji satu model tabular non-neural lain yang sesuai dengan aturan tugas. Jika prediksi OOF model tunggal berbeda secara berguna, uji blend probabilitas bersama CatBoost dan model ordinal. Pilih bobot blend dengan OOF saja.

**Lulus jika:** ensemble mengalahkan model tunggal pada rata-rata OOF, bukan hanya pada satu holdout.

### 7. Pilih, latih final, dan kirim submission

Pilih kandidat dengan rata-rata OOF terbaik dan variasi fold yang wajar. Latih ulang pada semua data train, simpan sebagai artefak dengan ID eksperimen baru, lalu buat submission sesuai urutan `building_id` di `sample_submission.csv`. Catat public score, tanggal, dan ID model di `reports/submission_log.csv`. Pertahankan baseline dan semua submission lama.

## Aturan keputusan dan pencatatan

- Ubah satu hipotesis utama setiap eksperimen; gunakan ID eksperimen unik.
- Metrik utama adalah Macro F1 OOF; sertakan F1 tiap kelas dan confusion matrix.
- Jangan memilih threshold, bobot blend, atau fitur dengan melihat private/public leaderboard.
- Promosikan model hanya jika kenaikannya terulang pada validasi yang sama dan tidak merusak performa kelas tertentu secara besar.
- Bila eksperimen tidak meningkatkan OOF, catat hasilnya dan lanjut ke hipotesis berikutnya; jangan mengganti baseline.
- Neural network dilarang oleh aturan tugas.

## Prioritas praktis

Mulai dari tahap 1, lalu tahap 2 sebelum mencoba kalibrasi, ordinal, fitur, dan ensemble. Ini menghindari pengeluaran waktu untuk metode lanjutan jika skor 0,53163 ternyata belum stabil atau CatBoost belum selesai belajar. Submission Kaggle baru dilakukan setelah kandidat lolos pemeriksaan OOF.
