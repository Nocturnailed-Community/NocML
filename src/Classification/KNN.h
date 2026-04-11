#ifndef NOC_ML_KNN_H
#define NOC_ML_KNN_H

#include "../Utils/MathUtils.h"

namespace NocML {

    class KNN {
    public:
        /**
         * @param samples Pointer to the samples array (flat array: [num_samples * dims])
         * @param labels Pointer to the labels array
         * @param num_samples Current number of samples
         * @param dims Number of features per sample
         * @param k Number of neighbors to consider
         */
        KNN(float* samples, int* labels, size_t num_samples, size_t dims, size_t k = 3)
            : _samples(samples), _labels(labels), _num_samples(num_samples), _dims(dims), _k(k) {}

        /**
         * Predict the label for a new input vector.
         */
        int predict(const float* input);

        void setK(size_t k) { _k = k; }

    private:
        float* _samples;
        int* _labels;
        size_t _num_samples;
        size_t _dims;
        size_t _k;
    };

} // namespace NocML

#endif // NOC_ML_KNN_H
