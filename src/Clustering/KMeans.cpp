#include "KMeans.h"

namespace NocML {

    void KMeans::run(const float* data, size_t num_samples, int* assignments, int max_iterations) {
        if (num_samples < _k) return;

        // Basic initialization: pick first k samples as centroids if not already initialized
        // Note: For better results, user should initialize centroids or we'd need a random generator.
        // We'll assume the user has called setCentroid or we'll just take the first k.
        
        for (int iter = 0; iter < max_iterations; iter++) {
            bool changed = false;

            // 1. Assignment step
            for (size_t i = 0; i < num_samples; i++) {
                int bestCluster = predict(&data[i * _dims]);
                if (assignments[i] != bestCluster) {
                    assignments[i] = bestCluster;
                    changed = true;
                }
            }

            if (!changed && iter > 0) break;

            // 2. Update step: recalculate centroids
            // We need a way to sum features. We'll use the centroids array temporarily as a sum buffer.
            float clusterCounts[_k];
            for(size_t j=0; j<_k; j++) clusterCounts[j] = 0;

            // Clear centroids to use as sum array
            for (size_t j = 0; j < _k * _dims; j++) _centroids[j] = 0;

            for (size_t i = 0; i < num_samples; i++) {
                int c = assignments[i];
                clusterCounts[c]++;
                for (size_t d = 0; d < _dims; d++) {
                    _centroids[c * _dims + d] += data[i * _dims + d];
                }
            }

            // Divide by counts
            for (size_t j = 0; j < _k; j++) {
                if (clusterCounts[j] > 0) {
                    for (size_t d = 0; d < _dims; d++) {
                        _centroids[j * _dims + d] /= clusterCounts[j];
                    }
                }
            }
        }
    }

    int KMeans::predict(const float* input) {
        float minDist = 3.4028235E+38; // FLT_MAX
        int bestCluster = 0;

        for (size_t j = 0; j < _k; j++) {
            float dist = euclideanDistance(input, &_centroids[j * _dims], _dims);
            if (dist < minDist) {
                minDist = dist;
                bestCluster = j;
            }
        }
        return bestCluster;
    }

} // namespace NocML
