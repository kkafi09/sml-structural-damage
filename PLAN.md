# Workflow proyek

## Alur analisis

Jalankan notebook EDA, pemodelan, dan submission secara berurutan.
Preprocessing menggunakan Pipeline dan ColumnTransformer. Model menggunakan CatBoost sebagai baseline,
serta stack XGBoost-LightGBM dengan logistic meta-model dan kalibrasi kelas.

## Pembagian data

Training 80% digunakan untuk model dasar, calibration 10% untuk tuning dan kalibrasi,
dan evaluation 10% untuk pengukuran Macro F1 serta metrik per kelas.

## Penggunaan

Kode utama tersedia pada `scripts/preprocess.py`, `scripts/workflow.py`, dan `scripts/predict.py`.
Jalankan `python -m scripts.workflow` untuk training dan `python -m scripts.predict` untuk prediksi.
Jalankan `python -m unittest discover -s tests -v` untuk pengujian.

## Penyimpanan

Model disimpan lokal pada `models/` dan diabaikan Git. Folder dipertahankan melalui `.gitkeep`.
CSV prediksi tersedia pada `submissions/`, laporan pada `reports/`, dan grafik pada `reports/figures/`.
