# Laporan Proyek Machine Learning - Cintha Hafrida Putri

## Domain Proyek

Proyek ini berfokus pada analisis ulasan dan penilaian produk di Amazon 
untuk mendapatkan wawasan terkait preferensi pelanggan dan popularitas produk. 
Dataset ini mencakup lebih dari 1.000 data penilaian dan ulasan dengan berbagai atribut produk, 
seperti:

Informasi Produk: Product ID, Name, Category, Discounted Price, Actual Price, Discount Percentage.

Umpan Balik Pelanggan: Ratings, Number of Ratings, Review Title, and Content.

Informasi Pengguna: User ID and Name.

Tujuan utama dari proyek ini adalah memahami preferensi pelanggan, mengidentifikasi kategori produk yang populer, 
dan mengeksplorasi hubungan antara skor penilaian, tingkat diskon, serta fitur produk lainnya.

Analisis produk dan ulasan pelanggan di platform e-commerce seperti Amazon sangat 
penting karena dapat memberikan wawasan mengenai perilaku konsumen dan tren pasar.
Dalam lingkungan bisnis yang kompetitif, memahami kebutuhan dan preferensi pelanggan menjadi 
faktor kunci dalam meningkatkan kepuasan, memperkuat loyalitas, serta mendorong penjualan. 
Namun, karena data produk sering kali berjumlah besar dan kompleks, menemukan pola dan tren 
secara manual menjadi sangat sulit. Oleh karena itu, pendekatan berbasis machine learning,
seperti clustering, menjadi solusi efektif untuk mengatasi permasalahan ini. Clustering 
memungkinkan identifikasi kelompok produk atau pelanggan dengan karakteristik yang serupa,
sehingga strategi pemasaran dapat disesuaikan secara spesifik.

## Business Understanding

### Problem Statements

Menjelaskan pernyataan masalah latar belakang:
- Bagaimana menganalisis pola perilaku rating dan ulasan konsumen terhadap produk-produk Amazon?
- Apakah ada hubungan antara harga diskon, rating, dan jumlah ulasan terhadap performa produk?

### Goals

Menjelaskan tujuan dari pernyataan masalah:
- Menganalisis dan memahami pola perilaku konsumen melalui rating dan ulasan untuk meningkatkan pengalaman berbelanja.
- Mengidentifikasi faktor-faktor yang mempengaruhi performa produk untuk optimasi strategi penetapan harga dan promosi
  
    ### Solution statements
    - Mengembangkan model clustering menggunakan algoritma K-Means dan DBSCAN untuk mengelompokkan produk berdasarkan karakteristik utama:
  Rating, Rating count, Discount percentage, Price ratio, Review length dan mengukur menggunakan metrik
evaluasi clustering silhoutte.

## Data Understanding
Dataset ini memiliki data 1465 Peringkat dan Ulasan Produk Amazon sesuai detailnya yang tercantum di situs web resmi Amazon
Jumlah baris : 1465
Jumlah Kolom : 16
Tipe data : Object (semua kolom)

### Variabel-variabel pada Amazon Sales dataset:
| Fitur                | Deskripsi                                         |
|----------------------|---------------------------------------------------|
| product_id           | ID Produk                                         |
| product_name         | Nama Produk                                       |
| category             | Kategori Produk                                   |
| discounted_price     | Harga Diskon Produk                               |
| actual_price         | Harga Asli Produk                                 |
| discount_percentage  | Persentase Diskon untuk Produk                    |
| rating               | Rating Produk                                     |
| rating_count         | Jumlah orang yang memberikan rating di Amazon     |
| about_product        | Deskripsi Produk                                  |
| user_id              | ID pengguna yang menulis ulasan untuk Produk      |
| user_name            | Nama pengguna yang menulis ulasan untuk Produk    |
| review_id            | ID ulasan pengguna                                |
| review_title         | Judul ulasan                                      |
| review_content       | Isi ulasan                                        |
| img_link             | Link Gambar Produk                                |
| product_link         | Link Resmi Produk di Situs Web                    |

Download datset : [https://www.kaggle.com/datasets/karkavelrajaj/amazon-sales-dataset/code](https://www.kaggle.com/datasets/karkavelrajaj/amazon-sales-dataset/data)

## Exploratory Data Analysis
- Distribusi variabel numerik menunjukkan: Rating memiliki distribusi normal dengan median sekitar 4.1
Rating count memiliki distribusi skewed right
Discount percentage memiliki distribusi multimodal
- Analisis korelasi menunjukkan hubungan antara: Harga diskon dan harga asli (korelasi positif kuat)
Rating dan rating count (korelasi lemah)


## Data Preparation
Penanganan Tipe Data
- **Mengkonversi kolom harga**: Mengubah kolom harga dari string ke float dengan menghapus simbol mata uang (₹) dan tanda koma.
- **Mengubah kolom discount_percentage**: Mengubah dari string ke float dengan menghapus simbol '%'.
- **Konversi rating dan rating_count**: Mengubah tipe data kolom rating dan rating_count ke format numerik.

Penanganan Missing Value
- **Mengisi missing value pada kolom rating**: Mengisi dengan nilai median karena rating menggunakan skala 1-5.
- **Mengisi missing value pada kolom rating_count**: Mengisi dengan median karena distribusinya skewed.
- **Alasan**: Penggunaan median lebih tepat daripada mean untuk data yang tidak terdistribusi normal.

Feature Engineering
- **Fitur baru 'price_ratio'**: Membuat fitur 'price_ratio' yang dihitung dengan rumus `price_ratio = actual_price / discounted_price`.
- **Fitur 'review_length'**: Menambahkan fitur yang dihitung dari panjang kolom review_content.
- **Alasan**: Fitur ini dibuat untuk menangkap hubungan antara harga dan memberikan insight lebih detail tentang ulasan.

Standardisasi
- **Menggunakan StandardScaler**: Menormalkan fitur numerik agar skala antar fitur seragam.
- **Alasan**: Standardisasi ini penting agar tidak ada fitur yang mendominasi dalam proses clustering.

## Modeling
**Pemodelan Clustering**

Dalam pemodelan ini, terdapat tiga model yang digunakan, yaitu **KMeans dengan fitur terpilih**, **KMeans dengan fitur terpilih dan fitur tambahan**, dan **DBSCAN dengan fitur terpilih**. Berikut adalah penjelasan tahapan dan parameter yang digunakan dalam masing-masing model:

Model 1: KMeans dengan Fitur Terpilih

**Standarisasi Data:**
- **Parameter**: `StandardScaler()`
- Data numerik (`rating`, `rating_count`, `discount_percentage`) dinormalisasi untuk memastikan bahwa semua fitur berada dalam skala yang sama, menghindari bias pada fitur dengan rentang yang lebih besar.

**K-Means Clustering:**
- **Parameter**: `n_clusters` dalam rentang 2 hingga 10 (uji berbagai jumlah kluster).
- Model KMeans dilatih menggunakan data yang telah distandarisasi, dan metrik evaluasi seperti inertia dan silhouette score dihitung untuk setiap jumlah kluster yang diuji.

Model 2: KMeans dengan Fitur Terpilih dan Fitur Tambahan

**Penambahan Fitur:**
- Fitur baru (`price_ratio` dan `review_length`) ditambahkan ke dataset untuk meningkatkan informasi yang tersedia bagi model.
  - `price_ratio`: Rasio antara harga aktual dan harga diskon.
  - `review_length`: Panjang konten ulasan.

**Standarisasi Data:**
- Proses standarisasi dilakukan lagi untuk fitur baru dan yang sudah ada.

**K-Means Clustering:**
- Sama seperti model sebelumnya, KMeans digunakan dengan rentang `n_clusters` yang sama untuk menemukan kluster yang optimal dan mengevaluasi dengan metrik yang relevan.

Model 3: DBSCAN dengan Fitur Terpilih

**Penskalaan Data:**
- Data distandarisasi menggunakan `StandardScaler()`.

**DBSCAN Clustering:**
- **Parameter**: `eps=0.5`, `min_samples=5`
- DBSCAN digunakan untuk mendeteksi kluster berdasarkan densitas data, dan model menghitung label kluster. Hasil evaluasi termasuk jumlah kluster yang terbentuk dan jumlah noise.

Kelebihan dan Kekurangan dari Setiap Algoritma

**KMeans**
Dalam konteks data yang berisi fitur numerik seperti `rating`, `rating_count`, `discount_percentage`, `price_ratio`, dan `review_length`, KMeans berfungsi dengan baik jika data tersebut tidak memiliki outlier signifikan dan jika kluster berbentuk bulat. Karena data yang digunakan mencakup fitur yang terkait langsung dengan produk dan penilaian, algoritma ini memberikan hasil yang relevan. Namun, jika terdapat banyak variasi dalam data (misalnya, produk dengan diskon ekstrem atau rating yang tidak biasa), hasil klustering mungkin tidak mencerminkan struktur data dengan akurat.

**DBSCAN**
Jika dataset memiliki produk dengan pola rating yang sangat bervariasi atau diskon yang tidak teratur, DBSCAN akan lebih efektif karena kemampuannya dalam mendeteksi kluster dengan bentuk tidak beraturan dan mengidentifikasi outlier. DBSCAN akan sangat berguna untuk dataset yang mengandung noise, misalnya produk dengan rating yang ekstrem, dan memberikan keunggulan dalam menemukan struktur kluster yang lebih halus dalam data.

**Pemilihan Model Terbaik**

Berdasarkan hasil evaluasi:
- **Model 1 KMeans (Fitur Terpilih)**: Silhouette Score: 0.6349
- **Model 2 KMeans (Fitur Terpilih + Fitur Tambahan)**: Silhouette Score: 0.8997
- **Model 3 DBSCAN**: Silhouette Score: 0.3555

**Model 2** dipilih sebagai model terbaik karena memiliki silhouette score tertinggi (0.8997), yang menunjukkan bahwa kluster yang dihasilkan lebih terpisah dengan baik dibandingkan dengan model lainnya. Hal ini mengindikasikan bahwa fitur tambahan meningkatkan pemisahan antar kluster.

**Cara Kerja dan Parameter dari Setiap Model**

**KMeans**
Menggunakan centroid untuk mengelompokkan data. Setiap iterasi, pusat kluster diperbarui hingga konvergensi. Parameter `n_clusters` menentukan jumlah kluster, sedangkan `random_state` digunakan untuk memastikan hasil yang konsisten.

**DBSCAN**
Mengelompokkan titik data berdasarkan densitas. `eps` menentukan jarak maksimum antara dua titik untuk menganggap mereka dalam kluster yang sama, sedangkan `min_samples` menentukan jumlah titik minimum dalam `eps` untuk membentuk kluster.

**Proses Pencarian K Optimal**

Untuk menemukan jumlah kluster optimal `k` dalam model KMeans, metode yang digunakan adalah dengan menghitung dan membandingkan nilai inertia dan silhouette score untuk setiap `k` dalam rentang yang telah ditentukan (2-10). Metrik ini digunakan untuk menentukan trade-off antara kompakness (inertia) dan pemisahan (silhouette score) dari kluster yang terbentuk, dengan tujuan untuk memilih nilai `k` yang memberikan hasil terbaik.

Secara umum, silhouette score yang lebih tinggi menunjukkan kluster yang lebih baik, sedangkan inertia yang lebih rendah menunjukkan kluster yang lebih kompak. Mencari `k` optimal adalah bagian penting dari proses modeling, dan dapat melibatkan metode lain seperti elbow method atau silhouette analysis.


## Evaluation
Metrik Evaluasi untuk Model Clustering
![alt text](https://github.com/cintha22059/PA_MLT/blob/main/plot%20silhouette%20score.png?raw=true)

Silhouette Score
- **Deskripsi**: Mengukur seberapa mirip objek dengan cluster-nya sendiri dibandingkan dengan cluster lain.
- **Range nilai**: -1 hingga 1.
- **Interpretasi**: Skor yang lebih tinggi menunjukkan cluster yang lebih baik.
- **Hasil**:
Komparasi Silhouette Score 
- Model 1 KMeans Selected features: 0.6349327834910277
- Model 2 KMeans Selected features dan Add new Features: 0.8997343627882889
- Model 3 DBSCAN selected features: 0.3554689722570361

Sehingga diputuskan model kluster terbaik adalah model 2 Kmeans dengan nilai Silhouette Score 0.89 dimana menerapkan features selection dan penambahan features.

Berikut merupakan hasil analisis Cluster dari model terbaik :

**Cluster 0:**

Review Length: Memiliki banyak outlier dengan review yang sangat panjang (hingga 17500 karakter), namun median sekitar 1000-1500 karakter

Price Ratio: Memiliki outlier hingga 16x lipat, dengan median sekitar 2x

Discount Percentage: Median diskon sekitar 50% dengan rentang 30-65%

Rating Count: Jumlah rating relatif rendah (median <50000)

Rating: Median rating sekitar 4.1 dengan beberapa outlier rendah hingga 2.0

Analisis:
Cluster 0 merepresentasikan produk dengan engagement pengguna yang lebih rendah namun memiliki review yang lebih detail. Produk-produk ini cenderung memiliki variasi harga yang ekstrem dan diskon yang moderat. Meski jumlah rating rendah, rating keseluruhan cukup baik dengan beberapa kasus ketidakpuasan.



**Cluster 1:**

Review Length: Lebih konsisten dengan sedikit outlier, median sekitar 1000-1500 karakter

Price Ratio: Lebih stabil dengan median sekitar 2x dan sedikit outlier

Discount Percentage: Distribusi diskon mirip dengan Cluster 0

Rating Count: Jumlah rating sangat tinggi (median >150000)

Rating: Rating lebih konsisten dengan median sekitar 4.2
Analisis:

Cluster 1 merepresentasikan produk populer dengan engagement tinggi. Review cenderung lebih singkat namun konsisten. Harga dan diskon lebih stabil dibanding Cluster 0. Produk-produk ini memiliki basis pengguna yang besar dengan rating yang konsisten tinggi, menunjukkan kualitas dan kepuasan pelanggan yang lebih terjamin.

**Evaluasi terhadap Business Understanding**
Hasil analisis cluster berhasil menjawab problem statement dengan memberikan wawasan mendalam mengenai perilaku konsumen dan faktor-faktor yang mempengaruhi performa produk. Cluster 0 dan Cluster 1 menunjukkan pola perilaku yang berbeda dalam hal panjang ulasan, rasio harga, diskon, jumlah rating, dan rating yang diberikan, yang mencerminkan preferensi konsumen terhadap berbagai jenis produk.

Model berhasil mencapai goals yang diharapkan. Analisis yang dilakukan pada masing-masing cluster memberikan pemahaman yang lebih baik tentang bagaimana konsumen berinteraksi dengan produk berdasarkan rating dan ulasan mereka. Selain itu, identifikasi cluster yang menunjukkan produk dengan engagement rendah dan tinggi dapat membantu dalam pengambilan keputusan strategis.

Solusi yang direncanakan berdampak signifikan. Dengan memisahkan produk ke dalam cluster berdasarkan karakteristik yang telah dianalisis, perusahaan dapat mengadopsi pendekatan yang lebih terarah dalam pemasaran dan penetapan harga. Misalnya, untuk Cluster 0, perusahaan dapat mempertimbangkan untuk meningkatkan interaksi dengan konsumen untuk produk dengan review panjang namun engagement rendah, sedangkan untuk Cluster 1, perusahaan dapat memanfaatkan popularitas dan kualitas produk untuk strategi promosi yang lebih agresif.

