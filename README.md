# Prediksi Harga Penjualan Mobil Bekas (USD)

Model *deep learning* (ANN regresi) untuk menaksir **harga jual mobil bekas** pada anak perusahaan dealer otomotif. Seluruh harga dinyatakan dalam **USD**.

---

## 1. Dataset

| Item | Keterangan |
|---|---|
| File | `1B.parquet` |
| Ukuran | 2.000 baris, 18 kolom |
| Target | `selling_price` (USD) |
| Duplikat | Tidak ada |
| Missing value | 2 baris pada `selling_price` |

**Fitur numerik:** `year`, `km_driven`, `engine`, `max_power`, `torque` (teks, diekstrak menjadi `torque_value` dan `rpm`)

**Fitur kategorikal:** `Region`, `State or Province`, `City`, `fuel`, `seller_type`, `transmission`, `owner`, `mileage`, `sold`, dan `seats` (diperlakukan sebagai kategori karena berupa pengelompokan jumlah kursi)

**Kolom dibuang:** `Sales_ID` dan `name` (identitas, tidak berpengaruh pada harga).

---

## 2. Exploratory Data Analysis (ringkasan temuan)

- `max_power` memiliki nilai **negatif** (min −100).
- `year` memiliki nilai anomali **3011**.
- `seats` didominasi nilai 5 (kuartil 75% = 5), namun ada nilai ekstrem hingga 14.
- `km_driven` dan `selling_price` memiliki banyak **outlier**; sebagian besar distribusi condong (*skewed*).
- `Region` memiliki nilai `N/A`, dan `transmission` memiliki nilai kosong (tanpa nama kategori).
- Kelas `sold`: N = 1.489, Y = 511.

---

## 3. Pre-processing

1. **Pemisahan data 70 : 10 : 20** (train : val : test) dengan `random_state=123`.
2. **Pembersihan anomali** (fungsi `clean_static_anomalies`, diterapkan terpisah ke train/val/test):
   - `year` 3011 → 2011
   - `max_power` negatif → nilai absolut
   - `transmission` kosong → `NaN`
   - Feature engineering: kolom `torque` diekstrak menjadi `torque_value` dan `rpm`
3. **Imputasi target** dengan median `y_train` (dihitung dari train saja agar tidak terjadi *data leakage*).
4. **Pipeline `ColumnTransformer`:**
   - Numerik: `SimpleImputer(median)` → `StandardScaler`
   - Kategorikal: `SimpleImputer(most_frequent)` → `OneHotEncoder(handle_unknown='ignore')`
   - `seats`: `SimpleImputer(most_frequent)` → `OrdinalEncoder`
   - `fit_transform` hanya pada train; val dan test memakai `transform`.

---

## 4. Model
### Baseline
```
Input(input_dim)
→ Dense(min_neurons, ReLU)
→ Dense(min_neurons, ReLU)
→ Dense(1, linear)
```
Adam, loss MSE, metrik MAE, 20 epoch, batch size 32.

### Modified
```
Input(input_dim)
→ Dense(min_neurons × 4, ReLU) → Dropout(0.2)
→ Dense(min_neurons × 2, ReLU) → Dropout(0.2)
→ Dense(min_neurons, ReLU)
→ Dense(1, linear)
```
Adam (learning rate 0.001), loss MSE, batch size 32, maksimal 100 epoch dengan `EarlyStopping` (monitor `val_loss`, patience 10, `restore_best_weights=True`). Training berhenti di epoch 24 dan bobot terbaik diambil dari epoch 14.

**Alasan modifikasi:** jaringan dibuat lebih lebar dan dalam agar dapat menangkap pola non-linear yang lebih kompleks; *Dropout* dan *EarlyStopping* ditambahkan untuk menekan overfitting.

---

## 5. Hasil Evaluasi (data test, harga dalam USD)

| Metrik | Baseline | Modified |
|---|---|---|
| RMSE | 2.209,88 | **1.911,13** |
| MAE | 1.220,13 | **1.080,29** |
| MAPE | 23,04% | **21,42%** |
| R² | 0,9399 | **0,9551** |

**Analisis singkat**

- Model *modified* lebih baik pada keempat metrik: rata-rata kesalahan absolut turun sekitar 140 USD dan R² naik menjadi ±0,955.
- Kurva pembelajaran model *modified* lebih stabil dan dimulai dari loss yang lebih rendah. Namun, masih ada selisih antara train loss dan validation loss, yang menandakan **overfitting ringan**.
- Berdasarkan heatmap korelasi, prediktor utama harga adalah `max_power`, `engine`, dan `year`, sedangkan `km_driven`, `owner`, dan `transmission` berpengaruh moderat/negatif.

---
