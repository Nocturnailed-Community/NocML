#include <NocML.h>

/**
 * NocML Real-World Example: Smart Agriculture (Plant Health Prediction)
 * 
 * Scenario: 
 * We have two sensors: an analog soil moisture sensor (A0) and a temperature sensor.
 * We want to predict if the plant needs watering based on a small dataset of known states.
 * 
 * Features:
 * 1. Preprocessing: We use MinMaxScaler to normalize raw sensor values.
 * 2. Classification: We use K-Nearest Neighbors (KNN) to classify the realtime data.
 */

const size_t NUM_SAMPLES = 4;
const size_t DIMS = 2; // [Moisture, Temperature]

// Normalized Training Data (0.0 to 1.0 range)
float trainingData[NUM_SAMPLES * DIMS] = {
  0.2, 0.8,  // Dry, Hot -> Class 1 (Needs Water)
  0.3, 0.7,  // Dry, Warm -> Class 1
  0.8, 0.3,  // Wet, Cool -> Class 0 (Healthy)
  0.7, 0.4   // Wet, Warm -> Class 0
};

// Labels corresponding to data
int labels[NUM_SAMPLES] = {1, 1, 0, 0}; 

// Initialize KNN model (K=3)
NocML::KNN knn(trainingData, labels, NUM_SAMPLES, DIMS, 3);

// Initialize Data Scaler
// Real-world sensor bounds: Moisture (0 to 1023), Temp (10C to 50C)
float minVals[DIMS] = {0.0, 10.0};
float maxVals[DIMS] = {1023.0, 50.0};
NocML::MinMaxScaler scaler(DIMS, minVals, maxVals);

void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println("--- Smart Agriculture Machine Learning System ---");
}

void loop() {
  // 1. Read Raw Sensors (Simulated limits here for demonstration)
  float rawMoisture = 300.0; // Assume we read analogRead(A0)
  float rawTemp = 38.0;      // Assume we read from a DHT22
  
  Serial.print("Raw Sensor -> Moisture: "); Serial.print(rawMoisture);
  Serial.print(" | Temp: "); Serial.println(rawTemp);
  
  float rawFeatures[DIMS] = {rawMoisture, rawTemp};
  float scaledFeatures[DIMS];
  
  // 2. Normalize Data using NocML Scaler
  scaler.transform(rawFeatures, scaledFeatures);
  
  Serial.print("Scaled Features -> ["); 
  Serial.print(scaledFeatures[0]); Serial.print(", ");
  Serial.print(scaledFeatures[1]); Serial.println("]");
  
  // 3. Classify with KNN
  int prediction = knn.predict(scaledFeatures);
  
  // 4. Result Action
  if (prediction == 1) {
    Serial.println("=> CLASSIFICATION: Plant needs water! Activating pump...\n");
  } else {
    Serial.println("=> CLASSIFICATION: Plant is healthy.\n");
  }
  
  delay(5000);
}
