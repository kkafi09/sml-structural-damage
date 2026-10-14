# Aturan tugas dan keputusan proyek

Sumber utama: [Tugas Supervised Machine Learning 2026](https://docs.google.com/document/d/1e8SbPqX3BX6_hsmawrtFu9kn80WotSfbORxYNg6cNBE/edit).
Dokumen dibaca saat audit. Ringkasan ini membedakan ketentuan pengajar, contoh struktur, dan keputusan proyek.
Aturan kompetisi pada halaman Kaggle tetap perlu dibaca setelah bergabung.

## Ketentuan eksplisit pengajar

- Jangan commit dataset besar di atas 100 MB.
- Gunakan `.gitignore` untuk `data/*.csv`; dokumentasikan cara unduh data dalam README.
- Gunakan pipeline dalam preprocessing, training, dan tuning.
- Setiap function/class di file `.py` harus memiliki docstring.
- Dilarang menggunakan neural network.
- Penilaian repository mencakup tree structure, workflow analisis, `Pipeline` dan `ColumnTransformer`, docstring, petunjuk penggunaan, dan interpretasi dalam README.
- Submission mengikuti kolom template dan disimpan dengan `index=False`.

Dokumen memberi contoh folder `data/`, `notebooks/`, `scripts/`, `models/`, `submissions/`, dan `reports/figures/`.
Nama file dalam contoh tidak berarti implementasi wajib identik.
Materi OvA/OvO merupakan referensi; dokumen tidak secara eksplisit mewajibkan perbandingan eksperimen keduanya.
Proyek ini tetap menyertakannya sebagai analisis tambahan.

Bobot penilaian: evaluasi publik 15%, evaluasi privat 35%, repository GitHub 50%.
Ringkasan ini tidak memuat klaim hasil atau peringkat proyek.

## Kontrak data dan submission

Target CSV aktual berbentuk `Grade 1`–`Grade 5`; model utama menormalkannya menjadi integer 1–5.
File prediksi memiliki tepat dua kolom:

```csv
building_id,damage_grade
122207000141,1
```

Jumlah, nilai, dan urutan ID harus sama dengan template dan test.
Tidak boleh ada duplikat ID, null, label di luar 1–5, atau kolom indeks tambahan.

## Keputusan analisis proyek

- Gunakan Macro F1, stratifikasi, evaluasi per kelas, dan confusion matrix.
- Tampilkan EDA, heatmap null, distribusi kelas, sentinel, outlier, dan hubungan fitur.
- Pelajari imputasi, vocabulary, encoding target, dan tetangga hanya dari training.
- Hitung encoding training secara cross-fit; calibration digunakan untuk tuning, meta-model, dan kalibrasi.
- Evaluation dan test tidak memilih parameter.
- OvA/OvO memakai estimator, fitur, subset, dan split yang sama.
- Simpan model serta kandidat baru terpisah dari CSV final.

Heatmap, split 80/10/10, seed, metrik lokal, dan perbandingan strategi merupakan keputusan proyek;
semuanya tidak diklaim sebagai kewajiban eksplisit dalam dokumen pengajar.

## Referensi dalam dokumen

Tutorial Kaggle, materi OvA/OvO, dan contoh repository [github.com/m-tsaqif](https://github.com/m-tsaqif).
