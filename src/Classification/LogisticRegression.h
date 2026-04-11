#ifndef NOC_ML_LOGISTIC_REGRESSION_H
#define NOC_ML_LOGISTIC_REGRESSION_H

#include <Arduino.h>
#include <math.h>

namespace NocML {

    class LogisticRegression {
    public:
        /**
         * @param dims Number of features
         * @param weights Pointer to weights array (size: dims)
         * @param bias Bias term (intercept)
         */
        LogisticRegression(size_t dims, const float* weights, float bias)
            : _dims(dims), _weights(weights), _bias(bias) {}

        /**
         * Predict the probability (0 to 1).
         */
        float predictProbability(const float* input) {
            float z = _bias;
            for (size_t i = 0; i < _dims; i++) {
                z += _weights[i] * input[i];
            }
            return 1.0f / (1.0f + exp(-z));
        }

        /**
         * Predict the class (0 or 1).
         * @param threshold Probability threshold (default 0.5)
         */
        int predict(const float* input, float threshold = 0.5f) {
            return predictProbability(input) >= threshold ? 1 : 0;
        }

    private:
        size_t _dims;
        const float* _weights;
        float _bias;
    };

} // namespace NocML

#endif // NOC_ML_LOGISTIC_REGRESSION_H
