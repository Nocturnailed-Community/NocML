#include <NocML.h>

/**
 * NocML Real-World Example: Smart Thermostat Forecasting
 * 
 * Scenario: 
 * A smart air conditioner wants to predict the room's temperature
 * slightly into the future to proactively adjust cooling power.
 * It uses a LinearForecaster to compute the trend of the last 5 minutes.
 */

const size_t HISTORY_SIZE = 5;

// Time records (e.g., minutes: 0, 1, 2, 3, 4)
float timeIndices[HISTORY_SIZE] = {0, 1, 2, 3, 4};

// Temperature records correspond to the timeIndices (Rising Trend)
float tempRecords[HISTORY_SIZE] = {24.0, 24.5, 25.1, 25.4, 26.0};

NocML::LinearForecaster forecaster;

void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println("--- Smart AC Temperature Forecaster ---");
  
  // 1. Fit historical data
  forecaster.fit(timeIndices, tempRecords, HISTORY_SIZE);
  
  Serial.println("Current History:");
  for (size_t i = 0; i < HISTORY_SIZE; i++) {
    Serial.print(" Minute "); Serial.print(timeIndices[i]);
    Serial.print(" -> "); Serial.print(tempRecords[i]); Serial.println(" C");
  }
  
  Serial.print("Temperature Trend (Slope): ");
  Serial.println(forecaster.getSlope());
  Serial.println();
}

void loop() {
  // 2. Predict Future Value (Next time-step, which is minute 5)
  float predictedNextTemp = forecaster.forecastNext();
  
  Serial.print("Forecast for Next Minute (5): ");
  Serial.print(predictedNextTemp);
  Serial.println(" C");
  
  // 3. Action based on forecast
  if (predictedNextTemp > 26.5) {
    Serial.println("=> FORECAST ALERT: Temp will exceed 26.5C. Activating AC preemptively!\n");
  } else {
    Serial.println("=> Temp trend is acceptable.\n");
  }
  
  // 4. Arbitrary Future Prediction
  float targetMinute = 10.0;
  Serial.print("Forecast for Minute 10: ");
  Serial.print(forecaster.predict(targetMinute));
  Serial.println(" C");
  
  delay(10000); // Wait before next cycle
}
