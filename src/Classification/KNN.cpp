#include "KNN.h"

namespace NocML {

    int KNN::predict(const float* input) {
        if (_num_samples == 0) return -1;

        // Arrays to keep track of the k-nearest distances and their labels
        float bestDistances[_k];
        int bestLabels[_k];

        // Initialize with infinity
        for (size_t i = 0; i < _k; i++) {
            bestDistances[i] = 3.4028235E+38; // FLT_MAX
            bestLabels[i] = -1;
        }

        for (size_t i = 0; i < _num_samples; i++) {
            float dist = euclideanDistance(input, &_samples[i * _dims], _dims);

            // Find if this distance is smaller than the current bests
            for (size_t j = 0; j < _k; j++) {
                if (dist < bestDistances[j]) {
                    // Shift existing bests to make room
                    for (size_t l = _k - 1; l > j; l--) {
                        bestDistances[l] = bestDistances[l - 1];
                        bestLabels[l] = bestLabels[l - 1];
                    }
                    bestDistances[j] = dist;
                    bestLabels[j] = _labels[i];
                    break;
                }
            }
        }

        // Voting system
        // Count frequencies of labels among the k neighbors
        // Note: For simplicity on Arduino, we assume labels are small positive integers
        // We'll just pick the one that appears most often.
        // If k=1, just return the first label.
        if (_k == 1) return bestLabels[0];

        int winnerLabel = -1;
        int maxVotes = 0;

        for (size_t i = 0; i < _k; i++) {
            if (bestLabels[i] == -1) continue;
            int currentLabel = bestLabels[i];
            int currentVotes = 0;
            for (size_t j = 0; j < _k; j++) {
                if (bestLabels[j] == currentLabel) currentVotes++;
            }
            if (currentVotes > maxVotes) {
                maxVotes = currentVotes;
                winnerLabel = currentLabel;
            }
        }

        return winnerLabel;
    }

} // namespace NocML
