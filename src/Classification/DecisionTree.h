#ifndef NOC_ML_DECISION_TREE_H
#define NOC_ML_DECISION_TREE_H

#include <Arduino.h>

namespace NocML {

    struct DecisionNode {
        int featureIdx;    // Index of the feature to split on. -1 if it's a leaf.
        float threshold;   // Threshold value for the split.
        int leftChild;     // Index of the left child in the nodes array.
        int rightChild;    // Index of the right child in the nodes array.
        int prediction;    // Leaf prediction (used only if featureIdx == -1).
    };

    class DecisionTreeClassifier {
    public:
        /**
         * @param nodes Pointer to an array of DecisionNode structs representing the tree.
         * @param numNodes Number of nodes in the array.
         */
        DecisionTreeClassifier(const DecisionNode* nodes, size_t numNodes)
            : _nodes(nodes), _numNodes(numNodes) {}

        /**
         * Predict class for input vector.
         */
        int predict(const float* input) {
            if (_numNodes == 0) return -1;
            
            int currentNodeIdx = 0;
            while (true) {
                const DecisionNode& node = _nodes[currentNodeIdx];
                
                // If it's a leaf node
                if (node.featureIdx == -1) {
                    return node.prediction;
                }
                
                // Traverse down
                if (input[node.featureIdx] <= node.threshold) {
                    currentNodeIdx = node.leftChild;
                } else {
                    currentNodeIdx = node.rightChild;
                }
                
                // Safety check to prevent infinite loops or out of bounds
                if (currentNodeIdx < 0 || (size_t)currentNodeIdx >= _numNodes) {
                    return -1;
                }
            }
        }

    private:
        const DecisionNode* _nodes;
        size_t _numNodes;
    };

} // namespace NocML

#endif // NOC_ML_DECISION_TREE_H
