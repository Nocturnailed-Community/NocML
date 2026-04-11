#include <NocML.h>

/**
 * NocML Real-World Example: Smart Home - AC Automation (Logistic Regression)
 * 
 * Scenario: 
 * We want to determine whether the Air Conditioner should be turned ON (Class 1)
 * or OFF (Class 0) based on two features: Room Temperature (C) and Humidity (%).
 * 
 * The model was trained in Python using Scikit-Learn's LogisticRegression 
 * and the weights/bias were exported here.
 */

// Model Parameters (Exported from Python's sklearn.linear_model.LogisticRegression)
const size_t DIMS = 2; // [Temperature, Humidity]
const float weights[DIMS] = {0.85, 0.42}; // Example weights
const float bias = -25.5;                 // Example bias

// Initialize Logistic Regression Model
NocML::LogisticRegression logReg(DIMS, weights, bias);

void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println("--- Smart Home AC Automation (Logistic Regression) ---");
}

void loop() {
  // 1. Simulate reading from DHT11 / DHT22 sensor
  float temperature = 28.5; // Celsius
  float humidity = 75.0;    // Percent
  
  float features[DIMS] = {temperature, humidity};
  
  Serial.print("Sensor -> Temp: "); Serial.print(temperature);
  Serial.print("C | Humidity: "); Serial.print(humidity); Serial.println("%");
  
  // 2. Predict probability & class
  float probability = logReg.predictProbability(features);
  int prediction = logReg.predict(features);
  
  // 3. Output result
  Serial.print("Probability of needing AC: ");
  Serial.print(probability * 100.0); Serial.println("%");
  
  if (prediction == 1) {
    Serial.println("=> ACTION: Turning ON Air Conditioner.\n");
  } else {
    Serial.println("=> ACTION: Turning OFF Air Conditioner.\n");
  }
  
  delay(5000);
}
