# NocML: Machine Learning for Arduino

[![Indonesian](https://img.shields.io/badge/Language-Indonesian-blue)](README_id.md)
[![GitHub Repository](https://img.shields.io/badge/GitHub-NocML-black?logo=github)](https://github.com/Nocturnailed-Community/NocML)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20092782.svg)](https://doi.org/10.5281/zenodo.20092782)

A lightweight, efficient C++ library for machine learning on microcontrollers, inspired by the **Scikit-Learn** API.

## 🚀 Key Features

- **Preprocessing**: `MinMaxScaler`, `StandardScaler`, `PolynomialFeatures`.
- **Classification**: `KNN`, `NaiveBayes`, `LogisticRegression`, `DecisionTreeClassifier`.
- **Clustering**: `K-Means`.
- **Forecasting**: `LinearForecaster` (Simple Linear Regression).
- **Optimization**: Zero dynamic memory allocation (`malloc`) during inference for maximum stability on low-resource boards like Arduino Uno/Nano.

---

## 🛠️ Installation

1. Download this repository.
2. Rename the folder to `NocML`.
3. Move it to your `Arduino/libraries/` folder.
4. Restart the Arduino IDE.

---

## 📖 Real-World Usage Guide

NocML is designed for real microcontroller limits. Below are complete real-world combinations of NocML features.

### 1. Smart Agriculture: Plant Health Prediction (KNN + Preprocessing)
**Scenario**: Predict if a plant needs water based on raw analog soil moisture and temperature readings.
<details>
<summary><b>View Code</b></summary>

```cpp
#include <NocML.h>

const size_t NUM_SAMPLES = 4;
const size_t DIMS = 2; // [Moisture, Temp]

// Normalized Training Data (0.0 - 1.0)
float trainingData[8] = {
  0.2, 0.8, // Dry, Hot -> Needs Water (1)
  0.3, 0.7, // Dry, Warm -> Needs Water (1)
  0.8, 0.3, // Wet, Cool -> Healthy (0)
  0.7, 0.4  // Wet, Warm -> Healthy (0)
};
int labels[4] = {1, 1, 0, 0}; 

NocML::KNN knn(trainingData, labels, NUM_SAMPLES, DIMS, 3);

// Scaler
float minVals[2] = {0.0, 10.0};
float maxVals[2] = {1023.0, 50.0};
NocML::MinMaxScaler scaler(DIMS, minVals, maxVals);

void setup() { Serial.begin(115200); }

void loop() {
  float raw[2] = {300.0, 38.0}; // Simulated readings
  float scaled[2];
  
  scaler.transform(raw, scaled);
  int prediction = knn.predict(scaled);
  
  if (prediction == 1) Serial.println("Plant needs water! Turning on pump...");
  delay(5000);
}
```
</details>

### 2. Predictive Maintenance: Motor Anomaly Detection (K-Means)
**Scenario**: Group unlabeled motor vibration data into "Normal" and "Abnormal" clusters.
<details>
<summary><b>View Code</b></summary>

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
  float currentVib = 5.7; // High vibration
  int clusterId = kmeans.predict(&currentVib);
  
  if (centroids[clusterId] > 3.0) {
    Serial.println("ANOMALY DETECTED! Maintenance required.");
  }
  delay(5000);
}
```
</details>

### 3. Smart AC Thermostat: Temperature Forecasting (Linear Regression)
**Scenario**: Predict future temperature index to proactively adjust AC cooling.
<details>
<summary><b>View Code</b></summary>

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
  float nextTemp = forecaster.forecastNext(); // Foresee minute 5
  
  if (nextTemp > 26.5) {
    Serial.println("FORECAST ALERT: Turning AC ON proactively.");
  }
  delay(10000);
}
```
</details>

### 4. Smart Home AC Automation (Logistic Regression)
**Scenario**: Fast binary decision (AC ON/OFF) based on exported parameters from Python.
<details>
<summary><b>View Code</b></summary>

```cpp
#include <NocML.h>

const float weights[2] = {0.85, 0.42}; // Exported from Sklearn
const float bias = -25.5;

NocML::LogisticRegression logReg(2, weights, bias);

void setup() { Serial.begin(115200); }

void loop() {
  float features[2] = {28.5, 75.0}; // [Temp, Humidity]
  int prediction = logReg.predict(features);
  
  if (prediction == 1) Serial.println("Turning ON Air Conditioner.");
  delay(5000);
}
```
</details>

### 5. Air Quality Classification (Naive Bayes)
**Scenario**: Identify safe vs hazardous air quality using Gaussian probability distributions, very fast for sensors.
<details>
<summary><b>View Code</b></summary>

```cpp
#include <NocML.h>

// Class 0: Good, Class 1: Hazardous
float means[4] = {100.0, 50.0, 400.0, 200.0}; // [C0_MQ135, C0_MQ7, C1_MQ135, C1_MQ7]
float vars[4]  = {400.0, 100.0, 1600.0, 400.0};
float priors[2] = {0.6, 0.4};

NocML::NaiveBayes nbModel(2, 2, means, vars, priors);

void setup() { Serial.begin(115200); }

void loop() {
  float features[2] = {350.0, 180.0}; // High values
  int prediction = nbModel.predict(features);
  
  if (prediction == 1) Serial.println("HAZARDOUS Air Quality! Alarm triggered.");
  delay(5000);
}
```
</details>

### 6. Object Sorting Robot (Decision Tree)
**Scenario**: Nested conditional logic exported directly from Scikit-Learn tree rules.
<details>
<summary><b>View Code</b></summary>

```cpp
#include <NocML.h>

// Feature 0: Height, Feature 1: Weight
NocML::DecisionNode tree[] = {
  {0, 10.0, 1, 2, -1}, // Root: if Height <= 10 go to 1, else 2
  {-1, 0, -1, -1, 0},  // Node 1: Leaf (Class 0 - Small)
  {-1, 0, -1, -1, 1}   // Node 2: Leaf (Class 1 - Large)
};

NocML::DecisionTreeClassifier treeModel(tree, 3);

void setup() { Serial.begin(115200); }

void loop() {
  float features[2] = {12.5, 120.0}; 
  int prediction = treeModel.predict(features);
  
  Serial.print("Sorting into Bin: ");
  Serial.println(prediction);
  delay(5000);
}
```
</details>

### 7. Full Synchronous Pipeline (Standard Scaler + Polynomials + Logistic Regression)
**Scenario**: Your raw sensor data requires complex non-linear transformation before classification. This shows how to chain multiple transformations synchronously right into a Machine Learning model.
<details>
<summary><b>View Code</b></summary>

```cpp
#include <NocML.h>

float means[1] = {500.0}; 
float stddevs[1] = {150.0};
NocML::StandardScaler scaler(1, means, stddevs);

NocML::PolynomialFeatures polyGen(3); // Degree 3 (x, x^2, x^3)

// Model expects 4 inputs after polynomial expansion
float weights[4] = {0.0, -1.5, 3.2, -0.8}; 
NocML::LogisticRegression logReg(4, weights, -2.1); 

void setup() { Serial.begin(115200); }

void loop() {
  float raw = 650.0; 
  
  // Pipeline Step 1: Scale
  float scaled[1];
  scaler.transform(&raw, scaled);
  
  // Pipeline Step 2: Expand
  float poly[4]; 
  polyGen.transform(scaled, 1, poly);
  
  // Pipeline Step 3: Classify
  int prediction = logReg.predict(poly);
  
  if (prediction == 1) Serial.println("Toxic Air Detected! (Class 1)");
  delay(5000);
}
```
</details>

---

## 🐍 Python Export Integration
To deploy complex models (e.g., Logistic Regression or Decision Trees) after training in Python, cleanly copy parameters via sklearn:

```python
import numpy as np
from sklearn.linear_model import LogisticRegression

model = LogisticRegression().fit(X, y)
print(f"Weights: {model.coef_[0]}")
print(f"Bias: {model.intercept_[0]}")
```

---

## ⚠️ Memory Management
- **Hardware Compatibility**: Optimized for AVR (Uno/Nano), ESP32, ESP8266, STM32, and ARM-based boards.
- **Resource Limits**: For 8-bit boards (Atmega328P), keep the number of samples and tree depth small to conserve SRAM. No `malloc` or `new` keyword runs during predictions.
- **Precision**: Uses `float` (32-bit) for all mathematical operations to balance speed and memory footprint.

---

## License
MIT License. Developed by Muhammad Ikhwan Fathulloh for NocML.
