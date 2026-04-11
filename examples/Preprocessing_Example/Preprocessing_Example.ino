#include <NocML.h>

/**
 * NocML Real-World Example: Synchronous Preprocessing & ML Pipeline
 * 
 * Scenario: 
 * We have an analog gas sensor measuring specific particles. The raw value 
 * does not have a linear boundary for classification. 
 * Therefore, our ML pipeline consists of 3 synchronous steps:
 * 1. StandardScaler: Normalize the raw sensor to Mean=0, Variance=1
 * 2. PolynomialFeatures: Expand the 1D scaled data to Degree 3 (x, x^2, x^3)
 * 3. LogisticRegression: Feed the expanded features (size 4) to classify 
 *    whether the air is "Safe" (0) or "Toxic" (1).
 */

const size_t DIMS = 1;

// 1. Scaler Parameters (Mean & StdDev from training data)
float means[DIMS] = {500.0}; 
float stddevs[DIMS] = {150.0};
NocML::StandardScaler scaler(DIMS, means, stddevs);

// 2. Polynomial Parameters (Degree 3)
const size_t DEGREE = 3;
NocML::PolynomialFeatures polyGen(DEGREE);
// Calculate output size: for 1D input, output is DEGREE + 1 = 4.
const size_t POLY_OUT_SIZE = 4;

// 3. Logistic Regression Parameters 
// Trained to accept 4 features: [bias/1, x, x^2, x^3]
const float weights[POLY_OUT_SIZE] = {0.0, -1.5, 3.2, -0.8}; 
const float bias = -2.1;
NocML::LogisticRegression logReg(POLY_OUT_SIZE, weights, bias);

void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println("--- Advanced Synchronous ML Pipeline ---");
}

void loop() {
  // A. Simulate raw sensor read
  float rawSensor = 650.0; // Assume we read analogRead(A0)
  Serial.print("1. Raw Sensor Value: "); Serial.println(rawSensor);
  
  // B. Pipeline Step 1: Standard Scaling
  float scaledSensor[DIMS];
  scaler.transform(&rawSensor, scaledSensor);
  Serial.print("2. Standardized Value: "); Serial.println(scaledSensor[0]);
  
  // C. Pipeline Step 2: Polynomial Expansion
  float polyFeatures[POLY_OUT_SIZE]; 
  polyGen.transform(scaledSensor, DIMS, polyFeatures);
  Serial.print("3. Expanded Features: [");
  for(int i = 0; i < POLY_OUT_SIZE; i++) {
    Serial.print(polyFeatures[i]);
    if(i < POLY_OUT_SIZE - 1) Serial.print(", ");
  }
  Serial.println("]");
  
  // D. Pipeline Step 3: Logistic Regression Inference
  int prediction = logReg.predict(polyFeatures);
  
  Serial.print("4. => FINAL RESULT: ");
  if (prediction == 1) {
    Serial.println("Toxic Air Detected! (Class 1)\n");
  } else {
    Serial.println("Safe Air (Class 0)\n");
  }
  
  delay(5000);
}
