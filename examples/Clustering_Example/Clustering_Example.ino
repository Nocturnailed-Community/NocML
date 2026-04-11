#include <NocML.h>

/**
 * NocML Real-World Example: Predictive Maintenance
 * 
 * Scenario: 
 * We are monitoring the vibration of a motor using an accelerometer.
 * over time, the motor might degrade and vibrate differently.
 * Without having labeled data of "broken" states ahead of time,
 * we use K-Means clustering to separate "Normal" operations from "Anomalies".
 */

const size_t K = 2; // Normal vs Abnormal
const size_t DIMS = 1; // 1D Data: Vibration Magnitude
const size_t NUM_SAMPLES = 12;

// Historic vibration data gathered from the motor
float vibrationData[NUM_SAMPLES] = {
  1.1, 1.2, 0.9, 1.3, 1.0, 1.2, // Normal operational vibration
  5.5, 6.2, 5.8, 6.0, 6.5, 5.9  // Abnormal/High vibration
};

// Storage for K-Means
float centroids[K * DIMS];
int assignments[NUM_SAMPLES];

NocML::KMeans kmeans(K, DIMS, centroids);

void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println("--- Motor Predictive Maintenance System ---");
  
  // 1. Train K-Means (10 iterations max)
  Serial.println("Training K-Means Model on historical data...");
  kmeans.run(vibrationData, NUM_SAMPLES, assignments, 10);
  
  Serial.print("Cluster 0 Center: "); Serial.println(centroids[0]);
  Serial.print("Cluster 1 Center: "); Serial.println(centroids[1]);
  Serial.println();
}

void loop() {
  // 2. Real-Time Inference
  // Simulate reading current vibration from I2C accelerometer
  float currentVibration = 5.7; 
  
  Serial.print("Current Vibration Reading: ");
  Serial.println(currentVibration);
  
  // predict which cluster this new reading belongs to
  int clusterId = kmeans.predict(&currentVibration);
  
  // Check if it belongs to the higher vibration cluster
  float normalThreshold = min(centroids[0], centroids[1]) + 2.0;

  if (centroids[clusterId] > normalThreshold) {
    Serial.println("=> ANOMALY DETECTED! Motor needs maintenance.\n");
  } else {
    Serial.println("=> Motor operating normally.\n");
  }
  
  delay(5000);
}
