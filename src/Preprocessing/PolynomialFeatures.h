#ifndef NOC_ML_POLYNOMIAL_FEATURES_H
#define NOC_ML_POLYNOMIAL_FEATURES_H

#include <Arduino.h>

namespace NocML {

    /**
     * PolynomialFeatures: Generates polynomial and interaction features.
     * Only supports Degree 2 for now to save memory on Arduino.
     */
    class PolynomialFeatures {
    public:
        /**
         * Calculate how many output features will be generated for a given input dimension.
         * For Degree 2: Original + Squared + Interactions + Bias(optional)
         */
        static size_t getOutputDims(size_t inputDims, bool includeBias = true) {
            // (n+d)! / (n!d!) - 1 (if no bias)
            // For d=2: 1 + n + n(n+1)/2
            size_t count = 1 + inputDims + (inputDims * (inputDims + 1) / 2);
            return includeBias ? count : count - 1;
        }

        /**
         * Transform input into polynomial features (Degree 2).
         */
        static void transform(const float* input, size_t inputDims, float* output, bool includeBias = true) {
            size_t idx = 0;
            
            if (includeBias) output[idx++] = 1.0f; // Bias

            // Linear terms
            for (size_t i = 0; i < inputDims; i++) {
                output[idx++] = input[i];
            }

            // Interaction and squared terms
            for (size_t i = 0; i < inputDims; i++) {
                for (size_t j = i; j < inputDims; j++) {
                    output[idx++] = input[i] * input[j];
                }
            }
        }
    };

} // namespace NocML

#endif // NOC_ML_POLYNOMIAL_FEATURES_H
