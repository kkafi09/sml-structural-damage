# Klasifikasi Kerusakan Bangunan

Proyek tugas Supervised Machine Learning ini memprediksi tingkat kerusakan bangunan setelah gempa Gorkha, Nepal. Target train berbentuk `Grade 1`–`Grade 5`; submission menggunakan angka 1–5. Evaluasi lokal memakai **Macro F1** agar setiap kelas memperoleh bobot yang sama.

Aturan pengajar tersedia di [rules.md](rules.md) dan status workflow di [PLAN.md](PLAN.md).

## Data dan instalasi

Gabung melalui [tautan kompetisi pengajar](https://www.kaggle.com/t/a9196b92680b4492aeb06aeaee94f654), terima aturan, lalu unduh `train.csv`, `test.csv`, `sample_submission.csv`, dan `metadata.csv` dari tab Data. Letakkan di `data/`. Dataset tidak disertakan dalam Git.

Gunakan Python 3.10 atau lebih baru. Contoh PowerShell di Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m ipykernel install --sys-prefix --name structural-damage --display-name "Python (structural-damage)"
.\.venv\Scripts\python.exe -m jupyter notebook
```

Pada Linux/macOS, gunakan `.venv/bin/python`. Pilih kernel **Python (structural-damage)**. Versi paket run terverifikasi tersedia di laporan workflow; `requirements-lock.txt` merekam environment yang digunakan.

## Urutan notebook

1. [01_eda.ipynb](notebooks/01_eda.ipynb): struktur dan metadata, null dan heatmap, duplikat, distribusi kelas, sentinel umur, histogram, boxplot, korelasi, dan hubungan kategori dengan target.
2. [02_modeling.ipynb](notebooks/02_modeling.ipynb): split stratified, baseline dan tuning terbatas, stack, metrik per kelas, confusion matrix, serta perbandingan OvA/OvO. Default menjalankan training penuh dari CSV mentah.
3. [03_submission.ipynb](notebooks/03_submission.ipynb): memuat model hasil training, membuat kandidat terpisah, serta memeriksa format, ID, label, dan distribusi prediksi.

Notebook berisi pemeriksaan, grafik, interpretasi, dan keputusan. Grafik disimpan di `reports/figures/`. Training menggunakan bagian data train yang telah ditentukan.

## Metode

Resep utama baru **official-data-stack-v1** hanya memakai CSV resmi. Preprocessing berada di [scripts/preprocess.py](scripts/preprocess.py), training di [scripts/workflow.py](scripts/workflow.py), dan inferensi di [scripts/predict.py](scripts/predict.py).

- Split 80% training, 10% calibration, 10% evaluation; stratified dengan seed 2026/2027.
- Feature engineering membuat rasio bangunan, indikator umur tidak diketahui, prefix survei, dan profil struktur. Kode wilayah merupakan kategori. Umur `999` diganti missing setelah indikator dibuat.
- `Pipeline` dan `ColumnTransformer` menggabungkan preprocessing dan estimator. Imputasi dan vocabulary kategori hanya dipelajari dari training; kategori baru dipetakan menjadi missing.
- Statistik target kelompok dan tetangga ID training dihitung 5-fold cross-fit. Calibration/evaluation/test hanya memakai statistik training tersimpan.
- Baseline memilih dua konfigurasi CatBoost pada calibration. Stack memakai tiga XGBoost dan tiga pasangan LightGBM multiclass/OvA dengan blend tetap 60/40.
- Logistic meta-model memakai enam blok log-probabilitas. Regularisasi dipilih melalui CV calibration; offset kelas dipilih dari probabilitas OOF calibration. Evaluation tidak memilih konfigurasi.
- OvA/OvO memakai SGDClassifier, fitur, seed, dan subset stratified 30.000 baris yang sama, serta evaluation yang sama. OvO melatih semua 10 pasangan, bukan hanya dua koreksi pasangan kelas.

Model dasar tetap memakai 80% training agar distribusi input meta-model sesuai calibration. Tidak dilakukan refit seluruh data yang mengubah resep tanpa evaluasi ulang. Seluruh preprocessing, model dasar, meta-model, dan offset disimpan bersama.

## Menjalankan workflow

Dari folder proyek setelah environment aktif:

```powershell
python -m scripts.workflow
python -m scripts.predict
python -m unittest discover -s tests -v
```

Training menghasilkan `models/reproducible_stack.joblib`, model baseline, `reports/workflow_validation.json`, dan prediksi evaluation. Inferensi menghasilkan `submissions/candidate_reproducible.csv`. Seluruh model biner dan cache preprocessing bersifat lokal, diabaikan Git, dan dapat dibuat ulang dari CSV mentah.

Untuk mengeksekusi notebook berurutan dari kernel bersih dan menyimpan output:

```powershell
python -m scripts.execute_notebooks
```

## Temuan data dan hasil

Train memiliki 533.465 baris dan test 228.629 baris. CSV mentah tidak memiliki null. Terdapat 2.198 umur sentinel `999` pada train dan 946 pada test; nilai ini ditangani sebagai umur tidak diketahui. Pemeriksaan rentang nilai dan kategori baru tetap diperlukan. Metadata menyebut flag `has_superstructure_*`, sedangkan CSV aktual memiliki `superstructure`.

Grade 1 berjumlah 55.170 (10,34%) dan Grade 5 berjumlah 193.036 (36,19%). Kelas terbesar sekitar 3,50 kali kelas terkecil. Laporan memakai Macro F1 bersama precision, recall, F1 per kelas, dan confusion matrix.

EDA dan enam pengujian kode sudah berhasil dijalankan. Jalankan notebook pemodelan dan submission untuk menghasilkan laporan evaluasi serta prediksi. Model disimpan secara lokal dan diabaikan Git.

Perbandingan baseline dan stack mengukur keseluruhan resep karena fitur dan estimator berbeda. OvA/OvO memakai kondisi sama untuk membandingkan strategi. Hubungan fitur pada EDA bersifat deskriptif, bukan bukti sebab-akibat.

## Submission dan struktur

Kandidat yang dihasilkan nantinya harus berisi `building_id,damage_grade`, satu label integer 1–5 per ID test, mengikuti template, tanpa indeks. Unggah kandidat yang dipilih secara manual melalui halaman kompetisi. Workflow tidak mengunggah atau mengganti CSV final lama otomatis.

```text
data/          CSV resmi lokal, diabaikan Git
notebooks/     EDA dengan output; pemodelan dan submission siap dijalankan
scripts/       alur utama modular dan eksperimen model
tests/         pengujian leakage, inferensi, dan format submission
models/        model lokal, seluruh model biner diabaikan Git
reports/       validasi dan figures/
submissions/   CSV prediksi dan submission
```
