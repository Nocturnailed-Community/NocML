#include <NocML.h>

/**
 * NocML Real-World Example: Air Quality Classification (Naive Bayes)
 * 
 * Scenario: 
 * We have two gas sensors: MQ-135 (Air Quality) and MQ-7 (Carbon Monoxide).
 * We want to classify the air quality into 2 classes: 0 (Good), 1 (Hazardous).
 * We use Gaussian Naive Bayes, which is extremely fast and suitable for 
 * independent continuous sensor readings.
 */

const size_t NUM_CLASSES = 2;
const size_t DIMS = 2; // [MQ-135, MQ-7]

// These represent the mean and variance of each feature per class.
// Computed during training phase (e.g., in Python).

// Group 1: Class 0 (Good Air)
// Group 2: Class 1 (Hazardous Air)
float means[NUM_CLASSES * DIMS] = {
  100.0, 50.0,   // Class 0: Low MQ-135, Low MQ-7
  400.0, 200.0   // Class 1: High MQ-135, High MQ-7
};

float variances[NUM_CLASSES * DIMS] = {
  400.0, 100.0,  // Class 0 Variances
  1600.0, 400.0  // Class 1 Variances
};

float classPriors[NUM_CLASSES] = {0.6, 0.4}; // We assume 60% of the time the air is Good

// Initialize Naive Bayes model
NocML::NaiveBayes nbModel(NUM_CLASSES, DIMS, means, variances, classPriors);

void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println("--- Air Quality Monitor (Gaussian Naive Bayes) ---");
}

void loop() {
  // 1. Simulate reading from analog gas sensors
  float mq135Val = 350.0; // Simulated moderately high value
  float mq7Val = 180.0;   // Simulated moderately high value
  
  float features[DIMS] = {mq135Val, mq7Val};
  
  Serial.print("Raw Sensor -> MQ-135: "); Serial.print(mq135Val);
  Serial.print(" | MQ-7: "); Serial.println(mq7Val);
  
  // 2. Classify
  int prediction = nbModel.predict(features);
  
  // 3. Output
  if (prediction == 0) {
    Serial.println("=> STATUS: Air Quality is Good. No action needed.\n");
  } else {
    Serial.println("=> STATUS: HAZARDOUS Air Quality! Triggering Exhaust Fan and Alarm...\n");
  }
  
  delay(5000);
}
