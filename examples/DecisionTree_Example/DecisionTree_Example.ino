#include <NocML.h>

/**
 * NocML Real-World Example: Object Sorting Robot (Decision Tree)
 * 
 * Scenario: 
 * A robotic arm needs to sort objects into 3 bins: Small (0), Medium (1), Large (2).
 * It uses a ToF distance sensor to measure 'Height' and a load cell to measure 'Weight'.
 * We have a exported Decision Tree model from Scikit-Learn that defines the rules.
 */

// Model exported from Python:
// node: {feature_idx, threshold, left_child_id, right_child_id, class_label}
// - If feature_idx == -1, it's a leaf node. class_label is used.
// - Features: [Height, Weight]

NocML::DecisionNode treeNodes[] = {
  // Node 0: Is Height <= 10.0 cm?
  {0, 10.0, 1, 2, -1}, 
  
  // Node 1 (Height <= 10): Is Weight <= 50.0 g?
  {1, 50.0, 3, 4, -1}, 
  
  // Node 2 (Height > 10): Is Weight <= 100.0 g?
  {1, 100.0, 5, 6, -1},
  
  // Leaves
  {-1, 0.0, -1, -1, 0}, // Node 3: Small
  {-1, 0.0, -1, -1, 1}, // Node 4: Medium
  {-1, 0.0, -1, -1, 1}, // Node 5: Medium
  {-1, 0.0, -1, -1, 2}  // Node 6: Large
};

const size_t NUM_NODES = 7;

// Initialize Decision Tree Classifier
NocML::DecisionTreeClassifier dtModel(treeNodes, NUM_NODES);

void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println("--- Object Sorting Robot (Decision Tree) ---");
}

void loop() {
  // 1. Simulate reading sensors
  float objectHeight = 12.5; // cm
  float objectWeight = 120.0; // grams
  
  float features[2] = {objectHeight, objectWeight}; // [Height, Weight]
  
  Serial.print("Scanned Object -> Height: "); Serial.print(objectHeight);
  Serial.print("cm | Weight: "); Serial.print(objectWeight); Serial.println("g");
  
  // 2. Classify through the tree
  int prediction = dtModel.predict(features);
  
  // 3. Act based on bin
  Serial.print("=> ACTION: Moving Object to Bin ");
  switch (prediction) {
    case 0: Serial.println("0 (SMALL Objects)\n"); break;
    case 1: Serial.println("1 (MEDIUM Objects)\n"); break;
    case 2: Serial.println("2 (LARGE Objects)\n"); break;
  }
  
  delay(5000);
}
