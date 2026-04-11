# NocML: Machine Learning untuk Arduino

[![English](https://img.shields.io/badge/Language-English-blue)](README.md)
[![GitHub Repository](https://img.shields.io/badge/GitHub-NocML-black?logo=github)](https://github.com/Nocturnailed-Community/NocML)

Library C++ yang ringan dan efisien untuk pemotongan machine learning di mikrokontroler, terinspirasi dari antarmuka API **Scikit-Learn**.

## 🚀 Fitur Utama

- **Preprocessing**: `MinMaxScaler`, `StandardScaler`, `PolynomialFeatures`.
- **Klasifikasi**: `KNN`, `NaiveBayes`, `LogisticRegression`, `DecisionTreeClassifier`.
- **Clustering**: `K-Means`.
- **Peramalan (Forecasting)**: `LinearForecaster` (Regresi Linear Sederhana).
- **Optimasi**: Tanpa alokasi memori dinamis (`malloc`) saat proses inferensi (prediksi) untuk stabilitas maksimum pada board dengan resource rendah seperti Arduino Uno/Nano.

---

## 🛠️ Instalasi

1. Download repositori ini.
2. Ganti nama folder menjadi `NocML`.
3. Pindahkan folder tersebut ke dalam direktori `Arduino/libraries/` Anda.
4. Restart Arduino IDE.

---

## 📖 Panduan Penggunaan Skala Nyata (Real-World)

NocML didesain dengan mempertimbangkan keterbatasan memori pada mikrokontroler. Berikut adalah kombinasi skenario penggunaan nyata (real-case) dari fitur-fitur NocML.

### 1. Pertanian Pintar: Prediksi Kesehatan Tanaman (KNN + Preprocessing)
**Skenario**: Memprediksi apakah tanaman butuh air berdasarkan kelembaban tanah analog mentah dan suhu lingkungan.
<details>
<summary><b>Lihat Kode</b></summary>

```cpp
#include <NocML.h>

const size_t NUM_SAMPLES = 4;
const size_t DIMS = 2; // [Kelembaban, Suhu]

// Data Latih Ternormalisasi (0.0 - 1.0)
float trainingData[8] = {
  0.2, 0.8, // Kering, Panas -> Butuh Air (1)
  0.3, 0.7, // Kering, Hangat -> Butuh Air (1)
  0.8, 0.3, // Basah, Dingin -> Sehat (0)
  0.7, 0.4  // Basah, Hangat -> Sehat (0)
};
int labels[4] = {1, 1, 0, 0}; 

NocML::KNN knn(trainingData, labels, NUM_SAMPLES, DIMS, 3);

// Timbangan (Scaler)
float minVals[2] = {0.0, 10.0};
float maxVals[2] = {1023.0, 50.0};
NocML::MinMaxScaler scaler(DIMS, minVals, maxVals);

void setup() { Serial.begin(115200); }

void loop() {
  float raw[2] = {300.0, 38.0}; // Simulasi pembacaan
  float scaled[2];
  
  scaler.transform(raw, scaled);
  int prediction = knn.predict(scaled);
  
  if (prediction == 1) Serial.println("Tanaman butuh air! Menyalakan pompa...");
  delay(5000);
}
```
</details>

### 2. Perawatan Prediktif: Deteksi Anomali Motor (K-Means)
**Skenario**: Mengelompokkan data getaran motor yang tak berlabel menjadi klaster "Normal" dan "Abnormal".
<details>
<summary><b>Lihat Kode</b></summary>

```cpp
#include <NocML.h>

const size_t K = 2; 
const size_t DIMS = 1; 
const size_t NUM_SAMPLES = 6;

float vibration[6] = {1.1, 1.2, 1.0, 5.5, 6.2, 5.9};
float centroids[2];
int assignments[6];

NocML::KMeans kmeans(K, DIMS, centroids);

void setup() {
  Serial.begin(115200);
  kmeans.run(vibration, NUM_SAMPLES, assignments, 10);
}

void loop() {
  float currentVib = 5.7; // Getaran tinggi
  int clusterId = kmeans.predict(&currentVib);
  
  if (centroids[clusterId] > 3.0) {
    Serial.println("ANOMALI TERDETEKSI! Membutuhkan perbaikan alat.");
  }
  delay(5000);
}
```
</details>

### 3. Termostat Pintar: Ramalan Temperatur (Regresi Linear)
**Skenario**: Memprediksi suhu ruangan masa depan untuk mengatur hembusan AC secara proaktif di awal.
<details>
<summary><b>Lihat Kode</b></summary>

```cpp
#include <NocML.h>

float timeIndices[5] = {0, 1, 2, 3, 4};
float tempRecords[5] = {24.0, 24.5, 25.1, 25.4, 26.0};

NocML::LinearForecaster forecaster;

void setup() {
  Serial.begin(115200);
  forecaster.fit(timeIndices, tempRecords, 5);
}

void loop() {
  float nextTemp = forecaster.forecastNext(); // Meramal masa depan menit 5
  
  if (nextTemp > 26.5) {
    Serial.println("PERINGATAN RAMALAN: Menyalakan pendingin AC secara proaktif.");
  }
  delay(10000);
}
```
</details>

### 4. Otomatisasi AC Smart Home (Logistic Regression)
**Skenario**: Keputusan biner super cepat (AC ON/OFF) berdasarkan hasil parameter yang dijahit dari Python.
<details>
<summary><b>Lihat Kode</b></summary>

```cpp
#include <NocML.h>

const float weights[2] = {0.85, 0.42}; // Diambil dari Sklearn
const float bias = -25.5;

NocML::LogisticRegression logReg(2, weights, bias);

void setup() { Serial.begin(115200); }

void loop() {
  float features[2] = {28.5, 75.0}; // [Suhu, Kelembaban]
  int prediction = logReg.predict(features);
  
  if (prediction == 1) Serial.println("Menyalakan AC Pendingin Ruangan.");
  delay(5000);
}
```
</details>

### 5. Klasifikasi Kualitas Udara (Naive Bayes)
**Skenario**: Mengidentifikasi kualitas udara sehat vs berbahaya memanfaatkan probabilitas Gaussian, komputasi yang sangat cocok bagi sensor.
<details>
<summary><b>Lihat Kode</b></summary>

```cpp
#include <NocML.h>

// Class 0: Bagus, Class 1: Bahaya
float means[4] = {100.0, 50.0, 400.0, 200.0}; // [C0_MQ135, C0_MQ7, C1_MQ135, C1_MQ7]
float vars[4]  = {400.0, 100.0, 1600.0, 400.0};
float priors[2] = {0.6, 0.4};

NocML::NaiveBayes nbModel(2, 2, means, vars, priors);

void setup() { Serial.begin(115200); }

void loop() {
  float features[2] = {350.0, 180.0}; // Nilai gas tinggi
  int prediction = nbModel.predict(features);
  
  if (prediction == 1) Serial.println("Udara BERBAHAYA! Menyalakan kipas pembuangan.");
  delay(5000);
}
```
</details>

### 6. Robot Penyortir Barang (Decision Tree)
**Skenario**: Logika kondisional berlapis yang diekspor langsung dari _rules_ Scikit-Learn.
<details>
<summary><b>Lihat Kode</b></summary>

```cpp
#include <NocML.h>

// Fitur 0: Tinggi, Fitur 1: Berat
NocML::DecisionNode tree[] = {
  {0, 10.0, 1, 2, -1}, // Node Induk: jika Tinggi <= 10 lanjut ke 1, selainnya 2
  {-1, 0, -1, -1, 0},  // Node 1: Ujung (Kelas 0 - Kecil)
  {-1, 0, -1, -1, 1}   // Node 2: Ujung (Kelas 1 - Besar)
};

NocML::DecisionTreeClassifier treeModel(tree, 3);

void setup() { Serial.begin(115200); }

void loop() {
  float features[2] = {12.5, 120.0}; 
  int prediction = treeModel.predict(features);
  
  Serial.print("Meletakkan ke Kotak: ");
  Serial.println(prediction);
  delay(5000);
}
```
</details>

### 7. Pipeline Sinkron Penuh (Standard Scaler + Polynomials + Logistic Regression)
**Skenario**: Data mentah sensor Anda memerlukan transformasi non-linear yang kompleks sebelum bisa diklasifikasikan. Layaknya pipa bertahap, Anda mengeksekusinya secara berurutan/sinkron masuk tepat ke dalam model ML.
<details>
<summary><b>Lihat Kode</b></summary>

```cpp
#include <NocML.h>

float means[1] = {500.0}; 
float stddevs[1] = {150.0};
NocML::StandardScaler scaler(1, means, stddevs);

NocML::PolynomialFeatures polyGen(3); // Derajat 3 (x, x^2, x^3)

// Model diprediksi akan menerima 4 variabel setelah ekspansi polinomial
float weights[4] = {0.0, -1.5, 3.2, -0.8}; 
NocML::LogisticRegression logReg(4, weights, -2.1); 

void setup() { Serial.begin(115200); }

void loop() {
  float raw = 650.0; 
  
  // Pipeline Tahap 1: Skala Ulang
  float scaled[1];
  scaler.transform(&raw, scaled);
  
  // Pipeline Tahap 2: Ekspansi Fitur
  float poly[4]; 
  polyGen.transform(scaled, 1, poly);
  
  // Pipeline Tahap 3: Klasifikasi Cepat
  int prediction = logReg.predict(poly);
  
  if (prediction == 1) Serial.println("Udara Beracun Terdeteksi! (Kelas 1)");
  delay(5000);
}
```
</details>

---

## 🐍 Ekspor & Integrasi Python
Untuk menjahit model kompleks sekelas Logistic Regression atau Decision Tree setelah melatihnya di Python, lakukan cara mudah berikut untuk menduplikasi ke C++:

```python
import numpy as np
from sklearn.linear_model import LogisticRegression

model = LogisticRegression().fit(X, y)
print(f"Weights: {model.coef_[0]}")
print(f"Bias: {model.intercept_[0]}")
```

---

## ⚠️ Manajemen Memori Komponen
- **Kecocokan Hardware**: Telah dioptimalkan untuk AVR (Uno/Nano), ESP32, ESP8266, STM32, dan arsitektur ARM lainnya.
- **Batasan Resource**: Jika menggunakan board 8-bit (Atmega328P), pertahankan jumlah dataset atau kedalaman tree serendah mungkin untuk menghemat SRAM. Tidak ada sintaks dinamis `malloc` atau `new` selama prediksi berjalan.
- **Presisi Komputasi**: Secara permanen menggunakan basis `float` (32-bit) di semua kalkulasi matematis agar presisi namun menghemat limitasi.

---

## Lisensi
MIT License. Dikembangkan oleh Muhammad Ikhwan Fathulloh untuk NocEngine.