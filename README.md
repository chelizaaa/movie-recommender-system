# Laporan Proyek Machine Learning - Cheliza Sriayu Simarsoit

## Daftar Isi

- [Project Overview](#project-overview)
- [Business Understanding](#business-understanding)
- [Data Understanding](#data-understanding)
- [Data Preprocessing](#data-preprocessing)
- [Data Preparation](#data-preparation)
- [Modeling](#modeling)
- [Evaluation](#evaluation)
- [Kesimpulan](#kesimpulan)
- [Referensi](#referensi)

## Project Overview

Pada proyek ini akan dibahas mengenai permasalahan mengenai sebuah sistem yang dapat memberikan rekomendasi film kepada para pengguna berdasarkan penilaian pengguna lainnya dan keterkaitan profil atau preferensi pengguna terhadap film yang telah diberikan penilaian. Sistem rekomendasi ini akan berhubungan dan berkaitan dengan perkembangan teknologi khususnya dalam sektor perekonomian, bisnis, dan hiburan.

![movie-cover](https://user-images.githubusercontent.com/77439245/197213211-5ffadc9e-eddc-42de-a9d4-d2a7b2e2f14b.jpg)

**Gambar 1. Ilustrasi Sistem Rekomendasi Film**

Dalam dunia digital sekarang ini, hampir semua masyarakat dunia termasuk Indonesia pasti sering menonton film. Bahkan sejak dahulu, ketika pertama kali jaringan kabel dan televisi ditemukan, manusia sudah mengenal yang namanya film dan dunia hiburan secara komersial melalui stasiun-stasiun siaran di televisi. Bedanya adalah dahulu menggunakan teknologi dalam bentuk gelombang analog, sedangkan pada jaman sekarang sudah beralih ke teknologi penyiaran secara digital. [[1]](https://www.researchgate.net/publication/346898118_PERKEMBANGAN_DAN_TRANSFORMASI_TEKNOLOGI_DIGITAL)

Perkembangan industri film saat ini sangat pesat, baik itu melalui siaran TV konvensional hingga secara digital. Di antaranya terdapat *series*, film pendek, dokumenter, *sci-fi*, dan lain-lain melalui TV ataupun bioskop. Umumnya seseorang menonton film sebagai suatu hiburan dikala bosan ataupun menjadi sebuah hobi bagi orang tersebut. [[2]](https://ejournal.upi.edu/index.php/JATIKOM/article/view/33208)

Persaingan industri film semakin ketat bahkan sejak pandemi COVID-19. Di mana banyak orang yang justru beralih dari televisi ke film digital. Misalnya seperti layanan *streaming* film melalui YouTube, Netflix, Disney Hotstar, Amazon Prime, dan lain-lain. Layanan tersebut pastinya mengandalkan pengguna merekan yang berlangganan layanan *streaming*, sehingga dibutuhkan sebuah sistem yang dapat meningkatkan pengalaman pengguna aplikasinya salah satunya adalah untuk menyarankan atau merekomendasikan film yang mungkin akan disukai oleh pengguna tertentu atau film yang relevan dengan riwayat film yang disukai atau ditonton oleh pengguna. [[2]](https://ejournal.upi.edu/index.php/JATIKOM/article/view/33208)

Sistem untuk memberikan saran film berdasarkan preferensi atau profil masing-masing orang tersebut dinamakan sistem rekomendasi. Sistem rekomendasi menggunakan data yang telah dikumpulkan baik berdasarkan data pengguna maupun data dari film itu sendiri. Misalnya ketika pelanggan menonton sebuah film dengan genre tertentu, maka sistem dapat memberikan rekomendasi film lain dengan genre yang sama. Selain itu, sistem juga dapat memberikan rekomendasi berdasarkan *rating* dan preferensi pengguna lain dengan ciri-ciri ataupun profil pengguna satu yang masih relevan, mirip atau sama. [[3]](https://j-ptiik.ub.ac.id/index.php/j-ptiik/article/view/9163)

## Business Understanding

### Problem Statements

Berdasarkan latar belakang di atas, maka diperoleh rumusan masalah pada proyek ini, yaitu:
1. Bagaimana cara melakukan tahap pra-pemrosesan data sebelum data tersebut dimasukkan ke dalam model *machine learning*?
2. Bagaimana cara melakukan tahap persiapan data *movies* dan *rating* sebelum digunakan untuk melatih model *machine learning* sistem rekomendasi?
3. Bagaimana cara membuat model *machine learning* untuk sistem rekomendasi film menggunakan data *rating* atau penilaian pengguna terhadap film?

### Goals

Berdasarkan rumusan masalah di atas, maka diperoleh tujuan dari proyek ini, yaitu:
1. Untuk melakukan tahap pra-pemrosesan data sebelum data tersebut dimasukkan ke dalam model *machine learning*.
2. Untuk melakukan tahap persiapan data sehingga data siap digunakan melatih model *machine learning* sistem rekomendasi.
3. Untuk membuat model *machine learning* dalam memberikan rekomendasi *movie* atau film terbaik sesuai dengan *rating* dan pengguna tersebut.

### Solution Statements

Berdasarkan rumusan masalah dan tujuan di atas, maka disimpulkan beberapa solusi yang dapat dilakukan untuk mencapai tujuan dari proyek ini, yaitu:
1. Tahap pra-pemrosesan data atau *data preprocessing*, yaitu dengan menggabungkan data *movies* dan *ratings* menjadi sebuah data baru berdasarkan kolom `movieId` pada masing-masing dataset *movies* dan *ratings*.

2. Tahap persiapan data atau *data preparation* dilakukan dengan menggunakan beberapa teknik persiapan data, yaitu:
   - Melakukan proses pengecekan data yang hilang atau *missing value* pada data *movies* dan *ratings*.
   - Melakukan proses pengecekan data duplikat pada data *movie* dan *ratings*.
   - Melakukan proses pemisahan data pada kolom jenis-jenis genre yang tergabung di data *movies* menjadi masing-masing genre terpisah.

3. Tahap membuat model *machine learning* yang dapat memberikan rekomendasi film kepada pengguna berdasarkan *rating* atau penilaian pengguna terhadap film tertentu. Tahap pembuatan model *machine learning* untuk sistem rekomendasi menggunakan pendekatan *content-based filtering recommendation* dan *collaborative filtering recommendation*.

   - **Content-based Filtering Recommendation**
     
     *Content-based filtering* adalah teknik merekomendasikan item yang mirip dengan item yang disukai pengguna di masa lalu. *Content-based filtering* mempelajari profil minat pengguna baru berdasarkan data dari objek yang telah dinilai pengguna. Algoritma ini bekerja dengan menyarankan item serupa yang pernah disukai di masa lalu atau sedang dilihat di masa kini kepada pengguna. Semakin banyak informasi yang diberikan pengguna, semakin baik akurasi sistem rekomendasi.
     
     - TF-IDF Vectorizer
       
       TF-IDF Vectorizer atau *Term Frequency - Inverse Document Frequency Vectorizer* digunakan untuk menemukan representasi fitur penting dari setiap kategori film. TF-IDF Vectorizer dari *library* scikit-learn akan melakukan vektorisasi nilai dengan menggunakan metode `fit_transform` dan `transform`, serta melakukan tokenisasi data secara langsung.
       
       Algoritma TF-IDF menggunakan rumus untuk menghitung bobot atau *weight* masing-masing dokumen terhadap kata kunci, yaitu, [[4]](https://www.researchgate.net/publication/336982602_IMPLEMENTASI_TERM_FREQUENCY_-INVERSE_DOCUMENT_FREQUENCY_TF-IDF_DAN_VECTOR_SPACE_MODEL_VSM_UNTUK_PENCARIAN_BERITA_BAHASA_INDONESIA)
       
       $$W_{dt} = tf_{dt} \times IDF_{t}$$
       
       Di mana:
       $d =$ dokumen ke-d
       $t =$ kata ke-t dari kata kunci
       $W_{dt} =$ bobot dokumen ke-d terhadap kata ke-t
       $tf_{dt} =$ banyaknya kata yang dicari pada sebuah dokumen
       $IDF =$ Inversed Document Frequency
       
       Penggunaan algoritma *Term Frequency - Inverse Document Frequency* (TF-IDF) dengan cara memberikan bobot hubungan suatu kata atau term terhadap dokumen. [[5]](https://journal.uinjkt.ac.id/index.php/ti/article/view/8623/0) Frekuensi kemunculan kata di dalam suatu dokumen yang diberikan akan menunjukkan tingkat kepentingan kata itu di dalam dokumen tersebut. Bobot kata semakin besar jika kata tersebut semakin sering muncul dalam suatu dokumen, dan akan semakin kecil jika muncul dalam banyak dokumen. [[4]](https://www.researchgate.net/publication/336982602_IMPLEMENTASI_TERM_FREQUENCY_-INVERSE_DOCUMENT_FREQUENCY_TF-IDF_DAN_VECTOR_SPACE_MODEL_VSM_UNTUK_PENCARIAN_BERITA_BAHASA_INDONESIA)
       
     - Cosine Similarity
       
       *Cosine similarity* merupakan suatu pengukuran untuk mencari tingkat derajat kesamaan antar dua buah vektor dalam ruang dimensi dari nilai cosinus. Metode pencarian derajat kesamaan menggunakan *cosine similarity* memiliki nilai akurasi yang cukup tinggi karena tidak berpengaruh pada panjang atau pendeknya suatu dokumen. [[5]](https://journal.uinjkt.ac.id/index.php/ti/article/view/8623/0)
       
       $$Similarity = cos(\theta) = \frac{A \cdot B}{\|A\| \|B\|} = \frac{\displaystyle\sum^{n}_{i=1} (wA_i \times wB_i)} {\sqrt{\displaystyle\sum^{n}_{i=1} (wA_{i})^{2} } \sqrt{\displaystyle\sum^{n}_{i=1} (wB_{i})^{2}} }$$
       
       Di mana:
       $A \cdot B =$ dot product antara vektor A dan vektor B
       $\|A\| =$ panjang vektor A
       $\|B\| =$ panjang vektor B
       $\|A\| \|B\| =$ cross product antara $\|A\|$ dan $\|B\|$
       $wA_i =$ bobot term pada query ke-i
       $wB_i =$ bobot term pada dokumen ke-i
       $i =$ jumlah term dalam kalimat
       $n =$ jumlah vektor
     
   - **Collaborative Filtering Recommendation**
     
     *Collaborative filtering* adalah teknik merekomendasikan item yang mirip dengan preferensi pengguna yang sama di masa lalu, misalnya berdasarkan penilaian film yang telah diberikan oleh seorang pengguna. Sistem akan merekomendasikan film berdasarkan riwayat penilaian pengguna tersebut terhadap film dan genrenya.
     
     Ide utama dari metode *collaborative filtering* ini adalah untuk mengeksploitasi informasi mengenai perilaku di masa lampau maupun pendapat dari suatu komunitas pengguna yang kemudian dimanfaatkan untuk melakukan prediksi item mana yang akan disukai atau menarik bagi seorang pengguna. [[6]](https://jurnal.uns.ac.id/itsmart/article/view/590) *Collaborative filtering* menggunakan matriks yang berisi *user-item rating* sebagai sebuah masukan atau *input*, sedangkan *output*-nya akan menghasilkan prediksi numerik seberapa besar tingkat kesukaan seseorang terhadap item yang direkomendasikan dan juga daftar n-item yang direkomendasikan oleh sistem. [[6]](https://jurnal.uns.ac.id/itsmart/article/view/590)

## Data Understanding

![dataset](https://user-images.githubusercontent.com/77439245/197213374-f3c2026b-afd3-41ec-a8db-6f6404477e20.png)

**Gambar 2. Kaggle Dataset Movie Recommender System**

*Dataset* yang digunakan dalam proyek ini adalah *dataset* yang diambil dari platform Kaggle. Berikut merupakan detail *dataset* yang digunakan.

**Tabel 1. Informasi Dataset**
|                         | Keterangan                                                                                                            |
|-------------------------|-----------------------------------------------------------------------------------------------------------------------|
| Sumber                  | [Kaggle Dataset: Movie Recommender System Dataset](https://www.kaggle.com/datasets/gargmanas/movierecommenderdataset) |
| *Usability*             | 8.24                                                                                                                  |
| Lisensi                 | [GPL 2](http://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html)                                                     |
| Penilaian/*Rating*      | Silver                                                                                                                |
| Jenis dan Ukuran Berkas | 2 csv (2.98 MB)                                                                                                       |
| Kategori                | Movies, Ratings, Business, Entertainment                                                                              |

Dari *dataset* tersebut, digunakan dua buah *file* csv, yaitu `movies.csv` dan `ratings.csv`.

Berdasarkan *dataset* tersebut, diketahui jumlah data *movies* berdasarkan atribut movieId adalah sebanyak 9742 data, dan terdapat sebanyak 610 jumlah *rating* berdasarkan atribut userId.

Kemudian dilakukan proses *Exploratory Data Analysis* (EDA) yang merupakan proses investigasi awal pada data untuk menganalisis karakteristik, menemukan pola, anomali, dan memeriksa asumsi pada data.

- **Movies**
  
  Berikut merupakan informasi deskripsi variabel kolom atau atribut dari *dataset* *movies*, yaitu banyak kolom, nama kolom, jumlah data masing-masing kolom, dan tipe datanya.
  
  **Tabel 2. Deskripsi Variabel Dataset Movies**
  | # | Column  | Non-Null Count | Dtype  |
  |---|---------|----------------|--------|
  | 0 | movieId | 9742 non-null  | int64  |
  | 1 | title   | 9742 non-null  | object |
  | 2 | genres  | 9742 non-null  | object |
  
  Berikut merupakan isi dari *dataset* *movies* yang menampilkan 5 data pertama film.
  
  **Tabel 3. Isi Dataset Movies**
  |   | movieId | title                              | genres                                          |
  |---|---------|------------------------------------|-------------------------------------------------|
  | 0 | 1       | Toy Story (1995)                   | Adventure\|Animation\|Children\|Comedy\|Fantasy |
  | 1 | 2       | Jumanji (1995)                     | Adventure\|Children\|Fantasy                    |
  | 2 | 3       | Grumpier Old Men (1995)            | Comedy\|Romance                                 |
  | 3 | 4       | Waiting to Exhale (1995)           | Comedy\|Drama\|Romance                          |
  | 4 | 5       | Father of the Bride Part II (1995) | Comedy                                          |
  
  Berikut merupakan deskripsi statistik dari *dataset* *movies* yang menampilkan jumlah data, rata-rata, standar deviasi, nilai minimal, nilai kuartil bawah atau Q1, kuartil tengah atau Q2 atau median, kuartil atas atau Q3, dan nilai maksimum.
  
  **Tabel 4. Deskripsi Statistik Dataset Movies**
  |       | movieId       |
  |-------|---------------|
  | count | 9742.000000   |
  | mean  | 42200.353623  |
  | std   | 52160.494854  |
  | min   | 1.000000      |
  | 25%   | 3248.250000   |
  | 50%   | 7300.000000   |
  | 75%   | 76232.000000  |
  | max   | 193609.000000 |
  
- **Ratings**
  
  Berikut merupakan informasi deskripsi variabel kolom atau atribut dari *dataset* *rating*, yaitu banyak kolom, nama kolom, jumlah data masing-masing kolom, dan tipe datanya.
  
  **Tabel 5. Deskripsi Variabel *Dataset* *Ratings***
  | # | Column    | Non-Null Count  | Dtype   |
  |---|-----------|-----------------|---------|
  | 0 | userId    | 100836 non-null | int64   |
  | 1 | movieId   | 100836 non-null | int64   |
  | 2 | rating    | 100836 non-null | float64 |
  | 3 | timestamp | 100836 non-null | int64   |
  
  Berikut merupakan isi dari *dataset* *ratings* yang menampilkan 5 data pertama film.
  
  **Tabel 6. Isi *Dataset* *Ratings***
  |   | userId | movieId | rating | timestamp |
  |---|--------|---------|--------|-----------|
  | 0 | 1      | 1       | 4.0    | 964982703 |
  | 1 | 1      | 3       | 4.0    | 964981247 |
  | 2 | 1      | 6       | 4.0    | 964982224 |
  | 3 | 1      | 47      | 5.0    | 964983815 |
  | 4 | 1      | 50      | 5.0    | 964982931 |
  
  Berikut merupakan deskripsi statistik dari *dataset* *ratings* yang menampilkan jumlah data, rata-rata, standar deviasi, nilai minimal, nilai kuartil bawah atau Q1, kuartil tengah atau Q2 atau median, kuartil atas atau Q3, dan nilai maksimum.
  
  **Tabel 7. Deskripsi Statistik *Dataset* *Ratings***
  |       | userId        | movieId       | rating        | timestamp    |
  |-------|---------------|---------------|---------------|--------------|
  | count | 100836.000000 | 100836.000000 | 100836.000000 | 1.008360e+05 |
  | mean  | 326.127564    | 19435.295718  | 3.501557      | 1.205946e+09 |
  | std   | 182.618491    | 35530.987199  | 1.042529      | 2.162610e+08 |
  | min   | 1.000000      | 1.000000      | 0.500000      | 8.281246e+08 |
  | 25%   | 177.000000    | 1199.000000   | 3.000000      | 1.019124e+09 |
  | 50%   | 325.000000    | 2991.000000   | 3.500000      | 1.186087e+09 |
  | 75%   | 477.000000    | 8122.000000   | 4.000000      | 1.435994e+09 |
  | max   | 610.000000    | 193609.000000 | 5.000000      | 1.537799e+09 |

## Data Preprocessing

Tahap *data preprocessing* adalah teknik yang digunakan untuk mengubah data mentah menjadi data yang bersih yang siap untuk digunakan pada proses selanjutnya.

Pada tahap ini dilakukan penggabungan *dataset* *movies* dan *ratings* menggunakan *library* Pandas *merge* pada kolom `movieId` dari ke dua *dataset* tersebut.

**Tabel 8. Penggabungan Data *Movies* dan *Ratings***
|   | movieId | title            | genres                                          | userId | rating | timestamp  |
|---|---------|------------------|-------------------------------------------------|--------|--------|------------|
| 0 | 1       | Toy Story (1995) | Adventure\|Animation\|Children\|Comedy\|Fantasy | 1      | 4.0    | 964982703  |
| 1 | 1       | Toy Story (1995) | Adventure\|Animation\|Children\|Comedy\|Fantasy | 5      | 4.0    | 847434962  |
| 2 | 1       | Toy Story (1995) | Adventure\|Animation\|Children\|Comedy\|Fantasy | 7      | 4.5    | 1106635946 |
| 3 | 1       | Toy Story (1995) | Adventure\|Animation\|Children\|Comedy\|Fantasy | 15     | 2.5    | 1510577970 |
| 4 | 1       | Toy Story (1995) | Adventure\|Animation\|Children\|Comedy\|Fantasy | 17     | 4.5    | 1305696483 |

## Data Preparation

Tahap *data preparation* merupakan proses transformasi data menjadi bentuk yang dapat diterima oleh model *machine learning* nanti. Proses *data preparation* yang dilakukan, yaitu membersihkan data *missing value*, melakukan pengecekan data duplikat, dan pemisahan genre pada *dataset* *movie*.

1. **Pengecekan Missing Value**

   Proses pengecekan data yang hilang atau *missing value* dilakukan pada masing-masing *dataset* *movies* dan *ratings*. Berdasarkan hasil pengecekan, ternyata tidak ada data yang hilang atau *missing value* dari *dataset* tersebut.
   
2. **Pengecekan Data Duplikat**

   Proses pengecekan data yang hilang atau *missing value* dilakukan pada masing-masing *dataset* *movies* dan *ratings*. Berdasarkan hasil pengecekan, ternyata tidak ada data yang hilang atau *missing value* dari *dataset* tersebut.
   
3. **Pemisahan Genre pada *Dataset* Movies**
   
   Proses pemisahan data genre-genre film yang tergabung ke dalam satu atribut dilakukan untuk mempermudah proses klasifikasi judul film dengan genre film untuk proses sistem rekomendasi.
   
   **Tabel 9. Pemisahan Data Genre pada *Dataset* *Movies***
   |       | movieId | title                               | genres    |
   |-------|---------|-------------------------------------|-----------|
   | 0     | 1       | Toy Story (1995)                    | Adventure |
   | 1     | 1       | Toy Story (1995)                    | Animation |
   | 2     | 1       | Toy Story (1995)                    | Children  |
   | 3     | 1       | Toy Story (1995)                    | Comedy    |
   | 4     | 1       | Toy Story (1995)                    | Fantasy   |
   | ...   | ...     | ...                                 | ...       |
   | 22079 | 193583  | No Game No Life: Zero (2017)        | Fantasy   |
   | 22080 | 193585  | Flint (2017)                        | Drama     |
   | 22081 | 193587  | Bungo Stray Dogs: Dead Apple (2018) | Action    |
   | 22082 | 193587  | Bungo Stray Dogs: Dead Apple (2018) | Animation |
   | 22083 | 193609  | Andrew Dice Clay: Dice Rules (1991) | Comedy    |

## Modeling

Tahap pengembangan model *machine learning* atau modeling sistem rekomendasi dilakukan untuk memberikan hasil rekomendasi film terbaik kepada pengguna tertentu berdasarkan *rating* atau penilaian pengguna terhadap film tersebut. Tahap modeling yang dilakukan menggunakan teknik pendekatan *content-based filtering recommendation* dan *collaborative filtering recommendation*.

1. **Content-based Filtering Recommendation**
   
   Beberapa tahap yang dilakukan untuk membuat sistem rekomendasi dengan pendekatan *content-based filtering* adalah TF-IDF Vectorizer, *cosine similarity*, dan pengujian sistem rekomendasi.
   
   - TF-IDF Vectorizer
     
     TF-IDF Vectorizer akan melakukan transformasi teks judul film menjadi bentuk angka berupa matriks. Ukuran matriks yang dihasilkan dari proses TF-IDF Vectorizer ini adalah sebesar 22084 baris data judul film dan 24 kolom data genre film.
     
     **Tabel 10. Hasil Matriks TF-IDF Vectorizer**
     | title | thriller | genres | mystery | noir | no | action | fi | drama | imax | western | ... | adventure | fantasy | war | sci | comedy | animation | children | horror | documentary | musical
     |-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|
     | Stalingrad (2013) | 0.0 | 0.00000 | 0.0 | 0.0 | 0.00000 | 0.0 | 0.000000 | 1.0 | 0.0 | 0.0 | ... | 0.0 | 0.0 | 0.0 | 0.000000 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
     | Carry on Cabby (1963) | 0.0 | 0.00000 | 0.0 | 0.0 | 0.00000 | 0.0 | 0.000000 | 0.0 | 0.0 | 0.0 | ... | 0.0 | 0.0 | 0.0 | 0.000000 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
     | The Amazing Screw-On Head (2006) | 0.0 | 0.00000 | 0.0 | 0.0 | 0.00000 | 0.0 | 0.000000 | 0.0 | 0.0 | 0.0 | ... | 1.0 | 0.0 | 0.0 | 0.000000 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
     | Stargate (1994) | 0.0 | 0.00000 | 0.0 | 0.0 | 0.00000 | 0.0 | 0.707107 | 0.0 | 0.0 | 0.0 | ... | 0.0 | 0.0 | 0.0 | 0.707107 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
     | Addiction, The (1995) | 0.0 | 0.00000 | 0.0 | 0.0 | 0.00000 | 0.0 | 0.000000 | 1.0 | 0.0 | 0.0 | ... | 0.0 | 0.0 | 0.0 | 0.000000 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
     | Ali Wong: Baby Cobra (2016) | 0.0 | 0.57735 | 0.0 | 0.0 | 0.57735 | 0.0 | 0.000000 | 0.0 | 0.0 | 0.0 | ... | 0.0 | 0.0 | 0.0 | 0.000000 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
     | Great Mouse Detective, The (1986) | 0.0 | 0.00000 | 0.0 | 0.0 | 0.00000 | 0.0 | 0.000000 | 0.0 | 0.0 | 0.0 | ... | 0.0 | 0.0 | 0.0 | 0.000000 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
     | Alone in the Dark (2005) | 0.0 | 0.00000 | 0.0 | 0.0 | 0.00000 | 0.0 | 0.000000 | 0.0 | 0.0 | 0.0 | ... | 0.0 | 0.0 | 0.0 | 0.000000 | 0.0 | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 |
     | Sum of All Fears, The (2002) | 1.0 | 0.00000 | 0.0 | 0.0 | 0.00000 | 0.0 | 0.000000 | 0.0 | 0.0 | 0.0 | ... | 0.0 | 0.0 | 0.0 | 0.000000 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
     | Carrie (2002) | 0.0 | 0.00000 | 0.0 | 0.0 | 0.00000 | 0.0 | 0.000000 | 1.0 | 0.0 | 0.0 | ... | 0.0 | 0.0 | 0.0 | 0.000000 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
     
   - *Cosine Similarity*
     
     *Cosine similarity* akan melakukan perhitungan derajat kesamaan dari data *movie* berdasarkan antar judul film. Ukuran matriks yang dihasilkan dari proses perhitungan *cosine similarity* ini adalah sebesar 22084 baris data judul film dan 22084 kolom data judul film juga.
     
     **Tabel 11. Hasil Matriks *Cosine Similarity***
     | title | Wrong Turn (2003) | Marine, The (2006) | Bounce (2000) | Still Walking (Aruitemo aruitemo) (2008) | Collateral (2004) |
     |----------------------------------------------------------------|-----|-----|-----|-----|-----|
     | Messengers, The (2007)                                         | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
     | Scanners (1981)                                                | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
     | Prairie Home Companion, A (2006)                               | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
     | Journey to the Center of the Earth (1959)                      | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
     | Loved Ones, The (2009)                                         | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
     | Rome, Open City (a.k.a. Open City) (Roma, città aperta) (1945) | 0.0 | 0.0 | 1.0 | 1.0 | 0.0 |
     | Collateral (2004)                                              | 1.0 | 1.0 | 0.0 | 0.0 | 1.0 |
     | Cherrybomb (2009)                                              | 0.0 | 0.0 | 1.0 | 1.0 | 0.0 |
     | Steamboy (Suchîmubôi) (2004)                                   | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
     | Amazing Grace (2006)                                           | 0.0 | 0.0 | 1.0 | 1.0 | 0.0 |
     
   - Hasil *Top-N Recommendation*
     
     Setelah dilakukan tahap TF-IDF Vectorizer dan Cosine Similarity, selanjutnya melakukan pengujian terhadap sistem rekomendasi dengan pendekatan *content-based filtering recommendation*. Hasil pengujian tersebut sebagai berikut,
     
     Diambil sebuah judul film yang dipilih oleh pengguna.
     
     **Tabel 12. Judul Film yang Dipilih Pengguna**
     |       | movieId | title        | genres |
     |-------|---------|--------------|--------|
     | 22080 | 193585  | Flint (2017) | Drama  |
     
     Berikut adalah hasil rekomendasi judul film berdasarkan genre yang sama.
     
     **Tabel 13. Hasil Rekomendasi *Content-based Filtering***
     |   | title                      | genres  |
     |---|----------------------------|---------|
     | 0 | Yankee Doodle Dandy (1942) | Drama   |
     | 1 | Yankee Doodle Dandy (1942) | Musical |
     | 2 | Rainmaker, The (1997)      | Drama   |
     | 3 | Boogie Nights (1997)       | Drama   |
     | 4 | Witness (1985)             | Drama   |
     
     Berdasarkan hasil rekomendasi di atas, dapat dilihat bahwa sistem yang dibuat berhasil memberikan rekomendasi beberapa judul film berdasarkan sebuah judul film, yaitu 'Flint (2017)' dan dihasilkan rekomendasi judul film dengan genre film yang sama, yaitu drama.

2. **Collaborative Filtering Recommendation**
   
   Beberapa tahap yang dilakukan untuk membuat sistem rekomendasi dengan pendekatan *collaborative filtering* adalah *data preparation*, pembagian *dataset* menjadi data latih dan data validasi, serta membangun model dan menguji sistem rekomendasi.
   
   - Data Preparation
     
     Tahap *data preparation* dilakukan dengan proses *encoding* fitur userId pada *dataset* *ratings* dan fitur movieId pada *dataset* *ratings* menjadi sebuah *array*. Lalu hasil *encoding* tersebut akan dilakukan pemetaan atau *mapping* fitur yang telah dilakukan *encoding* tersebut ke dalam *dataset* *ratings*.
     
     Berdasarkan hasil *encoding* dan *mapping* tersebut, diperoleh jumlah *user* sebesar 32, jumlah film sebesar 2427, nilai *rating* minimal sebesar 0.5, dan nilai *rating* maksimal yaitu 5.0.
     
   - Membagi Data Latih dan Data Validasi
     
     Tahap pembagian *dataset* atau *split* *dataset* diawali dengan mengacak *dataset* *ratings*, kemudian melakukan pembagian menjadi data latih (*training data*) dan data validasi (*validation data*), yaitu dengan rasio data latih banding data validasi sebesar 80:10.
     
     **Tabel 14. Hasil Pembagian Dataset**
     |      | userId | movieId | rating | timestamp  | user | movie |
     |------|--------|---------|--------|------------|------|-------|
     | 1501 | 15     | 5618    | 3.0    | 1510578001 | 14   | 762   | 
     | 2586 | 19     | 1717    | 3.0    | 965705195  | 18   | 1511  |
     | 2653 | 19     | 2040    | 2.0    | 965706728  | 18   | 1552  | 
     | 1055 | 8      | 364     | 5.0    | 839463546  | 7    | 471   |
     | 705  | 6      | 329     | 4.0    | 845553200  | 5    | 599   |
     | ...  | ...    | ...     | ...    | ...        | ...  | ...   |
     | 4426 | 28     | 3793    | 3.5    | 1234516011 | 27   | 228   |
     | 466  | 4      | 3160    | 4.0    | 964539121  | 3    | 414   |
     | 3092 | 20     | 3408    | 3.5    | 1054037151 | 19   | 421   |
     | 3772 | 22     | 68358   | 0.5    | 1268727244 | 21   | 1151  |
     | 860  | 6      | 981     | 3.0    | 845556567  | 5    | 712   |
     
   - Model Development dan Hasil Rekomendasi
     
     Berdasarkan model *machine learning* yang telah dibangun menggunakan layer *embedding* dan *regularizer*, serta *adam optimizer*, *binary crossentropy loss function*, dan metrik RMSE (*Root Mean Squared Error*), diperoleh hasil pengujian sistem rekomendasi film dengan pendekatan *collaborative filtering* sebagai berikut.
     
     **Hasil Rekomendasi Film dengan Pendekatan Collaborative Filtering**
     ```
     73/73 [==============================] - 0s 2ms/step
     Showing recommendations for users: 22
     ========================================
     Movie with high ratings from user
     ----------------------------------------
     Shawshank Redemption, The (1994) : Crime|Drama
     Forrest Gump (1994) : Comedy|Drama|Romance|War
     Blade Runner (1982) : Action|Sci-Fi|Thriller
     One Flew Over the Cuckoo's Nest (1975) : Drama
     Midnight Cowboy (1969) : Drama
     ----------------------------------------
     Top 10 movie recommendation
     ----------------------------------------
     Sound of Music, The (1965) : Musical|Romance
     E.T. the Extra-Terrestrial (1982) : Children|Drama|Sci-Fi
     Young Frankenstein (1974) : Comedy|Fantasy
     Fantasia (1940) : Animation|Children|Fantasy|Musical
     Last of the Mohicans, The (1992) : Action|Romance|War|Western
     Muppet Christmas Carol, The (1992) : Children|Comedy|Musical
     Lord of the Rings, The (1978) : Adventure|Animation|Children|Fantasy
     Who Framed Roger Rabbit? (1988) : Adventure|Animation|Children|Comedy|Crime|Fantasy|Mystery
     Notebook, The (2004) : Drama|Romance
     Blood Diamond (2006) : Action|Adventure|Crime|Drama|Thriller|War
     ```

     Berdasarkan hasil rekomendasi film di atas, dapat dilihat bahwa sistem rekomendasi mengambil pengguna acak (19), lalu dilakukan pencarian film dengan *rating* terbaik dari *user* tersebut.
     - Rear Window (1954) : **Mystery**|**Thriller**
     - Heathers (1989) : **Comedy**
     - Indiana Jones and the Last Crusade (1989) : **Action**|**Adventure**
     - Ferris Bueller's Day Off (1986) : **Comedy**
     - Who Framed Roger Rabbit? (1988) : **Adventure**|**Animation**|**Children**|**Comedy**|**Crime**|**Fantasy**|**Mystery**

     Selanjutnya, sistem akan menampilkan 10 daftar film yang direkomendasikan berdasarkan genre yang dimiliki terhadap data pengguna acak tadi. Dapat dilihat bahwa sistem merekomendasikan beberapa film dengan genre yang sama, seperti
     - Smoke (1995) : **Comedy**|**Drama**
     - Fantasia (1940) : **Animation**|**Children**|**Fantasy**|**Musical**
     - Amistad (1997) : **Drama**|**Mystery**
     - Producers, The (1968) : **Comedy**
     - The Lair of the White Worm (1988) : **Comedy**|**Horror**

## Evaluation

1. **Content-based Filtering Recommendation**
   
   Tahap evaluasi untuk sistem rekomendasi dengan *content-based filtering* dapat menggunakan metrik *precision*.
   
   $$precision = \frac{TP}{TP + FP}$$
   
   Di mana:
   $TP =$ *True Positive*; rekomendasi yang sesuai
   $FP =$ *False Positive*; rekomendasi yang tidak sesuai
   
   Berdasarkan hasil rekomendasi film dengan pendekatan *content-based filtering* dapat dilihat bahwa hasil yang diberikan oleh sistem rekomendasi berdasarkan film **Flint (2017)** dengan genre **Drama**, menghasilkan 5 rekomendasi judul film yang tepat, meskipun terdapat 1 rekomendasi film dengan genre **Musical** dikarenakan film dengan judul **Yankee Doodle Dandy (1942)** memiliki dua genre sekaligus. Tetapi secara keseluruhan sistem merekomendasikan film dengan tepat.
   
   $$precision = \frac{5}{5 + 0} = 100\%$$
   
   Dengan begitu, diperoleh nilai *precision* sebesar **100%**.
   
2. **Collaborative Filtering Recommendation**
   
   Tahap evaluasi untuk sistem rekomendasi dengan *collaborative filtering* menggunakan metrik RMSE (Root Mean Squared Error). Rumus untuk mencari nilai RMSE sebagai berikut,
   
   $$RMSE=\sqrt{\sum^{n}_{i=1} \frac{y_i - y\\_pred_i}{n}}$$
   
   Di mana:
   $n =$ jumlah *dataset*
   $i =$ urutan data dalam *dataset*
   $y_i =$ nilai yang sebenarnya
   $y_{pred} =$ nilai prediksi terhadap $i$
   
   Nilai RMSE dari sistem rekomendasi dengan pendekatan *collaborative filtering* cukup rendah, yaitu 0.1249 pada *Training RMSE*, dan 0.2445 pada *Validation RMSE*. Sedangkan untuk nilai *training loss* sebesar 0.5536, dan *validation loss* sebesar 0.6619.
   
   Berikut merupakan hasil visualisasi grafik RMSE dan loss pada masing-masing data latih dan data validasi.
   
   ![rmse](https://user-images.githubusercontent.com/77439245/197214188-cbfd8a15-8273-445a-8e3e-55a3a7290a27.jpg)
   
   **Gambar 3. Grafik RMSE Data Latih dan Validasi**
   
   ![loss](https://user-images.githubusercontent.com/77439245/197214201-e5c79b72-8a56-4d48-af27-8fe8077b99d2.jpg)
   
   **Gambar 4. Grafik *Loss* Data Latih dan Validasi**

## Kesimpulan

Dengan begitu, dapat disimpulkan bahwa sistem berhasil melakukan rekomendasi baik dengan pendekatan *content-based filtering* maupun *collaborative filtering*. *Collaborative filtering* membutuhkan data penilaian film dari pengguna, sedangkan pada *content-based filtering*, data *rating* tidak dibutuhkan karena sistem akan merekomendasikan berdasarkan konten film tersebut, yaitu genre.

## Referensi

[1] M. Danuri, "Perkembangan dan Transformasi Teknologi Digital", *INFOKAM*, no. 2, pp. 116-123, Sep. 2019, Retrieved from: https://www.researchgate.net/publication/346898118_PERKEMBANGAN_DAN_TRANSFORMASI_TEKNOLOGI_DIGITAL.

[2] E. R. Agustian, Munir, and E. P. Nugroho, "Sistem Rekomendasi Film Menggunakan Metode Collaborative Filtering dan K-Nearest Neighbors", *JATIKOM: Jurnal Aplikasi dan Teori Ilmu Komputer*, vol. 3, no. 1, pp. 18-21, Mar. 2020, Retrieved from: https://ejournal.upi.edu/index.php/JATIKOM/article/view/33208.

[3] M. Fajriansyah, P. P. Adikara, and A. W. Widodo, "Sistem Rekomendasi Film Menggunakan Content Based Filtering", *Jurnal Pengembangan Teknologi Informasi dan Ilmu Komputer*, vol. 5, no. 6, pp. 2188-2199, May 2021, https://j-ptiik.ub.ac.id/index.php/j-ptiik/article/view/9163.

[4] W. Priatna and J. S. Hidayat, "Implementasi Term Frequency - Inverse Document Frequency (TF-IDF) dan Vector Space Model (VSM) untuk Pencarian Berita Bahasa Indonesia", *Pelita Teknologi: Jurnal Ilmiah Informatika, Arsitektur dan Lingkungan*, vol. 14, no. 2, pp. 119-133, Sep. 2019, Retrieved from: https://www.researchgate.net/publication/336982602_IMPLEMENTASI_TERM_FREQUENCY_-INVERSE_DOCUMENT_FREQUENCY_TF-IDF_DAN_VECTOR_SPACE_MODEL_VSM_UNTUK_PENCARIAN_BERITA_BAHASA_INDONESIA.

[5] V. Amrizal, "Penerapan Metode Term Frequency Inverse Document Frequency (TF-IDF) Dan Cosine Similarity Pada Sistem Temu Kembali Informasi Untuk Mengetahui Syarah Hadits Berbasis Web (Studi Kasus: Hadits Shahih Bukhari-Muslim)", *Jurnal Teknik Informatika*, vol. 11, no. 2, Okt. 2018 , Retrieved from: https://journal.uinjkt.ac.id/index.php/ti/article/view/8623/0.

[6] L. Dzumiroh and R. Saptono, "Penerapan Metode Collaborative Filtering Menggunakan Rating Implisit pada Sistem Perekomendasi Pemilihan Film di Rental VCD", *ITSMART: Jurnal Teknologi dan Informasi*, vol. 1, no. 2, Des. 2012, Retrieved from: https://jurnal.uns.ac.id/itsmart/article/view/590.
