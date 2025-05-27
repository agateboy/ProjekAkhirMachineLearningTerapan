# Laporan Proyek Machine Learning - Adrian Ramdhany

## Domain Proyek

Faktor lingkungan fisik merupakan faktor eksternal yang dapat mempengaruhi kinerja di tempat kerja, meliputi temperature, kelembaban udara, pencahayaan dan kebisingan yang memiliki peran signifikan dalam menciptakan suasana kerja optimal. [[1]](https://repository.uinjkt.ac.id/dspace/handle/123456789/34324) 

Banyaknya pekerja yang mengeluh tentang tidak sesuainya suara kebisingan dan cahaya sering mengganggu kesehatan setelah melakukan aktivitas pekerjaan. [[2]](https://doi.org/10.1007/s13762-016-1072-x)

Dalam proyek ini, saya menggunakan dataset yang telah diambil dari pembacaan nilai sensor menggunakan mikrokontroler dan nilai kenyamanan dari para pekerja sebanyak 2000 data dan disimpan dalam [Google Drive](drive.google.com).

Predictive alaytics memungkinkan perusahaan untuk mnganalisis tingkat kenyamanan ruangan berdasarkan data yang telah diperoleh dari berbagai ruangan dan testimoni para pekerja. Hal ini membantu pengambilan keputusan perusahaan dalam membuat standar kenyamanan pada setiap ruangan dalam meningkatkan kualitas pekerjaan para pekerja. Dalam proyek ini terdapat 3 model Machine Learning yang akan dibangun, yaitu K-Nearest Neighbors (KNN)*, *Gradient Boosting (GB)*, dan *Random Forest*. Dengan model ini dapat dilakukan pendekatan lebih mendalam tentang analisis kenyamanan lingkungan pekerja dari berbagai macam faktor yang ada.

## Business Understanding

Lingkungan kerja yang nyaman merupakan faktor penting dalam meningkatkan produktivitas, konsentrasi, dan kesejahteraan karyawan. Namun, tidak semua ruang kerja memiliki kondisi lingkungan yang mendukung kenyamanan tersebut. Variabel seperti suhu, kelembapan, kebisingan, pencahayaan, dan kadar oksigen dapat memengaruhi tingkat kenyamanan kerja seseorang.

Dalam konteks ini, perusahaan membutuhkan pendekatan berbasis data untuk memahami pengaruh faktor-faktor lingkungan terhadap kenyamanan kerja. Dengan solusi prediktif, perusahaan dapat melakukan penyesuaian atau perbaikan pada ruang kerja secara proaktif berdasarkan data sensor lingkungan yang dikumpulkan secara real-time.

### Problem Statements

- Bagaimana pengaruh faktor lingkungan seperti suhu, kelembapan, kebisingan, pencahayaan, dan oksigen terhadap tingkat kenyamanan kerja karyawan?
- Dapatkah sistem klasifikasi berbasis Machine Learning memprediksi tingkat kenyamanan kerja secara akurat berdasarkan data sensor lingkungan?

### Goals

- Mengidentifikasi dan menganalisis hubungan antara variabel lingkungan (suhu, kelembapan, kebisingan, pencahayaan, dan oksigen) dengan tingkat kenyamanan kerja.
- Membangun model klasifikasi untuk memprediksi tingkat kenyamanan pekerja dari data sensor secara akurat dan sesuai untuk ruang kerja.
- Menyediakan sistem pendukung keputusan berbasis data untuk pengelola fasilitas perusahaan dalam menciptakan ruang kerja yang optimal bagi karyawan.

## Solution statements
- Menerapkan tiga algoritma klasifikasi yaitu *K-Nearest Neighbors (KNN)*, *Gradient Boosting (GB)*, dan *Random Forest* pada model.
- Membandingkan tiga algoritma klasifikasi yaitu *K-Nearest Neighbors (KNN)*, *Gradient Boosting (GB)*, dan *Random Forest* pada model.
- Melakukan evaluasi model berdasarkan metrik accuracy, precision, recall dan f1-score.
- Memilih model dengan performa terbaik untuk diterapkan sebagai sistem prediktif tingkat kenyamanan ruangan di perusahaan.

## Data Understanding
Dataset yang digunakan dalam proyek ini merupakan hasil pengukuran berbagai sensor lingkungan yang dipasang di ruang kerja untuk memantau kondisi secara real-time. Data ini juga dilengkapi dengan skor kenyamanan (Comfort) dari warga perusahaan, yang dinilai dalam skala 1 hingga 5. Tujuannya adalah untuk mempelajari hubungan antara kondisi lingkungan dan persepsi kenyamanan.

### Sumber Dataset
Data diperoleh dari sensor-sensor yang terhubung oleh microcontroller ESP32 dengan mengumpulkan nilai-nilai lingkungan secara real-time.
- Temperature dan Humidity menggunakan sensor DHT22
- Noise menggunakan analog microphone (dB)
- Light menggunakan sensor lux
- Sensor oksigen menggunakan MQ-8
Data ini bersifat internal dan tidak diambil dari sumber publik. Tidak ada tautan sumber eksternal karena data dikumpulkan secara mandiri dari eksperimen.

Dataset ini terdapat jumlah bari 2000 dan memiliki 6 variabel dengan keterangan sebagai berikut.

Variabel|Keterangan|
------|------|
Temperature|Suhu udara dalam derajat Celcius|
Humidity|Kelembaban udara dalam persen|
Noise|Tingkat kebisingan dalam desibel|
Light|Intensitas cahaya dalam lux|
Oxygen|Kadar oksigen dalam persen|
Comfort|Skor kenyamanan yang diberikan (1-5)|

### 1. Deskripsi Variabel
Column      | Non-Null Count | Dtype  
------      | -------------- |-----  
Temperature | 2000 non-null  |float64
Humidity    | 2000 non-null  |float64
Noise       | 2000 non-null  |float64
Light       | 2000 non-null  |float64
Oxygen      | 2000 non-null  |float64
Comfort     | 2000 non-null  |int64 
Berdasarkan output di atas terdapat 5 kolom dengan tipe data float64 dan 1 kolom dengan tipe data int64

### 2. Pengecekan Nilai Null
` `|0
------|------
Temperature|0|
Humidity|0|
Noise	|0|
Light	|0|
Oxygen	|0|
Comfort	|0|

dtype: int64

- Missing values: Tidak ditemukan nilai yang hilang di seluruh kolom.
- Data duplikat: Tidak ditemukan baris duplikat.

### 3. Visualisasi data
Outliers merupakan sample yang nilainya jauh dari data utama dan hasil pengamatan dan muncul sangat jarang serta berbeda dengan hasil yang lainnya.

![1](https://github.com/user-attachments/assets/526aa8ef-7824-4149-b978-7769b6be69c1)

### 4. Menangani Outliers

Outliers ditangani menggunakan metode IQR (Interquartile Range) yang dapat menangani distribusi data outliers dengan baik.

```
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
outliers = ((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR)))
```

Berikut merupakan hasil data setelah dilakukan penanganan outliers menggunakan metode IQR.

![2](https://github.com/user-attachments/assets/05f80cbd-b673-49b2-80b6-fdc708a7ed82)

### 5. Univariative Multivariative Analysis
`Univariative` merupakan analisis distribusi masing-masing fitur seperti Temperature, Humidity, Noise, Light, Oxygen dan Comfort secara individu.

![3](https://github.com/user-attachments/assets/5110b499-9c25-45f1-adf5-014ad5d45bbf)

`Multivariative` melibatkan hubungan antar fitur seperti scatter matrix antar fitur untuk melihat pola clustering atau outliers.

![4](https://github.com/user-attachments/assets/d4bfd57f-ef48-42ea-bcda-8e2d858b4e0c)

### 6. Correlation Matrix

Pengecekan korelasi antar fitur menggunakan `heatmap correlation matrix`

![5](https://github.com/user-attachments/assets/b86f87a2-8989-4fd1-b629-ae08afc90dca)

Correlation matrix ini menunjukkan hubungan antara berbagai variabel seperti Temperature, Humidity, Noise, Light, Oxygen, dan Comfort. Nilai di dalam matriks adalah koefisien korelasi, yang berkisar dari -0.2 hingga 1:
- 1 menunjukkan korelasi positif sempurna.
- -0.2 menunjukkan korelasi negatif sempurna.

Dengan hasil:
- Noise memiliki korelasi negatif yang signifikan dengan Comfort (-0.37) yaitu semakin berisik lingkungan maka semakin renda tingkat kenyamanan.
- Light memiliki korelasi positif dengan Comfort (0.15) yaitu semakin baik pencahayaan mungkin meningkatkan kenyamanan.

- Oxygen berkorelasi positif dengan Comfort(0.16) mengindikasikan kadar oksigen yang lebih tinggi mungkin berkontribusi pada kenyamanan.


## Data Preparation

Tahap ini bertujuan untuk mempersiapkan data agar siap digunakan dalam proses pelatihan model Machine Learning. Proses yang dilakukan mencakup pembagian dataset dan standarisasi fitur numerik.

### Pemisahan Fitur dan Label

Pertama, data dipisahkan menjadi fitur (X) dan label (y). Label target yang ingin diprediksi adalah `Comfort`, sedangkan fitur lainnya digunakan sebagai variabel input model.

```python
X = kenyamanan_prediction.drop('Comfort', axis=1)
y = kenyamanan_prediction['Comfort']
```
### Split data, Train dan Test Set

Dataset dibagi menjadi data pelatihan (80%) dan data pengujian (20%) menggunakan train_test_split. Ini dilakukan agar model dapat dilatih pada sebagian data dan diuji performanya pada data yang belum pernah dilihat sebelumnya.

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

dengan total seluruh sampel dalam dataset adalah 2000, total sampel dalam train dataset adalah 1600 dan total sampel dalam test dataset adalah 400.

### Standarisasi Data

Karena model Machine Learning seperti KNN dan Gradient Boosting sensitif terhadap skala fitur, maka dilakukan standarisasi menggunakan `StandardScaler`. Standarisasi ini dilakukan hanya pada fitur numerik: Temperature, Humidity, Noise, Light, dan Oxygen.

Fitting scaler dilakukan hanya pada data pelatihan, lalu transformasi diterapkan ke data pelatihan dan pengujian agar tidak terjadi kebocoran data (data leakage).

```python
scaler.fit(X_train[numerical_features])
```

## Modeling
Pada tahap ini, terdapat tiga algpritma yang akan digunakan dalam membuat Machine Learning untuk melakukan analisis prediksi.

Dilakukan persiapan dataframe untuk menganalisis algoritma K-Nearest Neighboor (KNN), Gradient Boosting dan Random Forest.

#### 1. Model Development dengan K-Nearest Neighboor (KNN)

KNN bekerja dengan membandingkan jarak satu sampel ke sampel pelatihan lain dengan memilih sejumlah k tetangga terdekat dengan parameter `n-neighboors` dengan nilai `k=5`.

```python
knn = KNeighborsClassifier(n_neighbors=5)
```

#### 2. Model Development dengan Gradient Boosting

Gradient Boosting (GB) merupakan teknik Machine Learning berbasis ensemble yang membangun model dengan menggabungkan beberapa pohon keputusan secara bertahap.
`n_estimators=100` menentukan jumlah pohon keputusan yang akan dibangun dengan `learning_rate=0.1` yaitu menentukan seberapa besar kontribusi setiap pohon terhadap hasil akhir.


```python
gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
```

#### 3. Model Development dengan Random Forest

Random Forest merupakan algoritma ensemble berbasis decision tree yang menggabungkan banyak pohon untuk meningkatkan akurasi dan mengurangi overfitting. Algoritma ini efektif dalam klasifikasi karena tahan terhadap data yang tidak terstruktur dan memiliki kemampuan untuk menangani banyak fitur.

```python
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10, n_jobs=-1)
```

Parameter `n_estimator` dengan jumlah `100` pohon (trees), `max-depth` dengan nilai `10`, dan `n-jobs` yang bernilai `-1` yaitu pekerjaan dilakukan secara paralel.

## Evaluation


Dari model development KNN didapatkan confusion matrix sebagai berikut:

![9](https://github.com/user-attachments/assets/c708bdab-1174-41b7-9e8f-ccdbee1657d1)

Model Development dengan KNN menghasilkan accuracy dengan nilai `61,67%`

Dari model development Gradient Boosting didapatkan confusion matrix sebagai berikut:

![10](https://github.com/user-attachments/assets/4cdc2b48-7d58-4684-ac33-f88feaf79608)

Model development dengan GB menghasilkan accuracy dengan nilai `92,38%`

Dari model development Random Forest didapatkan confusion matrix sebagai berikut:

![11](https://github.com/user-attachments/assets/9547a143-a33c-47bb-ba79-f6cdad627d5f)

Model development dengan RF menghasilkan accuracy sebesar `95,17%`

Dari seluruh akurasi yang didapat dari keempat model, terdapat bar plot untuk melihat perbandingan nilai akurasi masing-masing model sebgai berikut.

![12](https://github.com/user-attachments/assets/aad66835-a206-40be-abbf-90647c6e0cd7)

| Model                     | Accuracy   | Precision | Recall   | F1-Score |
| ------------------------- | ---------- | --------- | -------- | -------- |
| K-Nearest Neighbors (KNN) | 61.67%     | 0.62      | 0.61     | 0.61     |
| Gradient Boosting (GB)    | 92.38%     | 0.92      | 0.92     | 0.92     |
| Random Forest (RF)        | **95.17%** | **0.95**  | **0.95** | **0.95** |

Berdasarkan gambar di atas dan evaluasi masing-masing model untuk mengetahui skor akurasi, F1 score, didapatkan model **Random Forest** merupakan model terbaik karena memiliki accuracy score dan F1 score tertinggi serta kesalahan klasifikasi yang paling sedikit.

## Kesimpulan

Berdasarkan hasil yang diperoleh setelah melakukan EDA dan pengujian model terbaik, dapat disimpulkan bahwa:
- Terdapat hubungan positif antara kadar oksigen dengan skor kenyamanan.
- Temperature dan Humidity memiliki hubungan yang signifikan dan saling mempengaruhi.
- Faktor kebisingan dan pencahayaan memiliki kontribusi yang lebih rendah secara statistik, namun tetap penting untuk menjaga kenyamanan ruangan.
- Proses pembersihan data outliers menggunakan IQR membantu meningkatkan kualitas data.
- Analisis univariate dan multivariate memberikan pemahaman mendalam tentang distribusi data dan hubungan antar variabel.

Model klasifikasi yang akurat dan andal sangat penting untuk membantu perusahaan memahami kondisi lingkungan kerja yang memengaruhi kenyamanan karyawan. Dengan Random Forest yang terbukti sebagai model terbaik:

- Perusahaan dapat mengotomatisasi pemantauan kenyamanan secara real-time berdasarkan input sensor.

- Sistem ini dapat digunakan sebagai sistem pendukung keputusan untuk perbaikan fasilitas (pencahayaan, ventilasi, akustik).

- Dapat diterapkan dalam pengembangan standar kenyamanan ruangan berbasis data.

## Referensi

[[1]](https://repository.uinjkt.ac.id/dspace/handle/123456789/34324) Sari, O. A. P. (2016). Hubungan Lingkungan Kerja Fisik dengan Kelelahan Kerja pada Kolektor Gerbang Tol Cililitan PT Jasa Marga Cabang Cawang Tomang Cengkareng tahun 2016 (Bachelor's thesis, FKIK UIN Jakarta).

[[2]](https://doi.org/10.1007/s13762-016-1072-x) Jafari, M., Mollasadeghi, A., Kargarsharifabad, H., & Rashedi, V. (2017). The effect of noise pollution on human health: A case study. International Journal of Environmental Science and Technology, 14(4), 687-694.

[[3]](https://doi.org/10.1177/1477153510367972) Boyce, P. R., Hunter, C., & Howlett, O. (2011). The Benefits of Daylight through Windows. Lighting Research & Technology, 43(2), 133-143.
