#ifndef NOC_ML_NAIVE_BAYES_H
#define NOC_ML_NAIVE_BAYES_H

#include <Arduino.h>
#include <math.h>

namespace NocML {

    class NaiveBayes {
    public:
        /**
         * @param num_classes Number of classes
         * @param dims Number of features
         * @param means Pointer to the array for storing class means (size: classes * dims)
         * @param variances Pointer to the array for storing class variances (size: classes * dims)
         * @param priors Pointer to class priors (size: classes)
         */
        NaiveBayes(size_t num_classes, size_t dims, float* means, float* variances, float* priors)
            : _num_classes(num_classes), _dims(dims), _means(means), _variances(variances), _priors(priors) {}

        /**
         * "Train" the model by setting the calculated parameters.
         * Users would typically do this offline and just upload the means/variances.
         */
        void updateParameters(size_t classIdx, const float* classMeans, const float* classVars, float prior) {
            if (classIdx >= _num_classes) return;
            _priors[classIdx] = prior;
            for (size_t i = 0; i < _dims; i++) {
                _means[classIdx * _dims + i] = classMeans[i];
                _variances[classIdx * _dims + i] = classVars[i];
            }
        }

        /**
         * Predict the class label.
         */
        int predict(const float* input) {
            float maxLogPost = -1.0e30; // Very small
            int bestClass = -1;

            for (size_t c = 0; c < _num_classes; c++) {
                float logPost = log(_priors[c]);
                for (size_t i = 0; i < _dims; i++) {
                    float mean = _means[c * _dims + i];
                    float var = _variances[c * _dims + i];
                    if (var < 1e-6) var = 1e-6; // Avoid division by zero

                    float diff = input[i] - mean;
                    // Gaussian log-likelihood
                    logPost -= 0.5 * log(2 * M_PI * var);
                    logPost -= (diff * diff) / (2 * var);
                }

                if (logPost > maxLogPost) {
                    maxLogPost = logPost;
                    bestClass = c;
                }
            }
            return bestClass;
        }

    private:
        size_t _num_classes;
        size_t _dims;
        float* _means;
        float* _variances;
        float* _priors;
    };

} // namespace NocML

#endif // NOC_ML_NAIVE_BAYES_H
