#ifndef NOC_ML_MATH_UTILS_H
#define NOC_ML_MATH_UTILS_H

#include <Arduino.h>
#include <math.h>

namespace NocML {

    /**
     * Calculate Euclidean distance between two vectors.
     */
    inline float euclideanDistance(const float* v1, const float* v2, size_t dims) {
        float sum = 0;
        for (size_t i = 0; i < dims; i++) {
            float diff = v1[i] - v2[i];
            sum += diff * diff;
        }
        return sqrt(sum);
    }

    /**
     * Calculate Squared Euclidean distance (Skipping sqrt() for massive optimization).
     * Ideal for comparisons in KNN and K-Means.
     */
    inline float squaredEuclideanDistance(const float* v1, const float* v2, size_t dims) {
        float sum = 0;
        for (size_t i = 0; i < dims; i++) {
            float diff = v1[i] - v2[i];
            sum += diff * diff;
        }
        return sum;
    }

    /**
     * Calculate Manhattan distance between two vectors.
     */
    inline float manhattanDistance(const float* v1, const float* v2, size_t dims) {
        float sum = 0;
        for (size_t i = 0; i < dims; i++) {
            sum += abs(v1[i] - v2[i]);
        }
        return sum;
    }

} // namespace NocML

#endif // NOC_ML_MATH_UTILS_H
