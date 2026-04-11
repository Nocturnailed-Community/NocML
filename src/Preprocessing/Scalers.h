#ifndef NOC_ML_SCALARS_H
#define NOC_ML_SCALARS_H

#include <Arduino.h>

namespace NocML {

    /**
     * MinMaxScaler: Rescales data to [0, 1] (or custom range).
     * X_std = (X - X_min) / (X_max - X_min)
     */
    class MinMaxScaler {
    public:
        MinMaxScaler(size_t dims, float* minValues, float* maxValues)
            : _dims(dims), _minValues(minValues), _maxValues(maxValues) {}

        void transform(const float* input, float* output) {
            for (size_t i = 0; i < _dims; i++) {
                float range = _maxValues[i] - _minValues[i];
                if (abs(range) < 1e-6) output[i] = 0;
                else output[i] = (input[i] - _minValues[i]) / range;
            }
        }

    private:
        size_t _dims;
        float* _minValues;
        float* _maxValues;
    };

    /**
     * StandardScaler: Rescales data to mean=0 and variance=1.
     * z = (x - u) / s
     */
    class StandardScaler {
    public:
        StandardScaler(size_t dims, float* means, float* stdDevs)
            : _dims(dims), _means(means), _stdDevs(stdDevs) {}

        void transform(const float* input, float* output) {
            for (size_t i = 0; i < _dims; i++) {
                if (abs(_stdDevs[i]) < 1e-6) output[i] = 0;
                else output[i] = (input[i] - _means[i]) / _stdDevs[i];
            }
        }

    private:
        size_t _dims;
        float* _means;
        float* _stdDevs;
    };

} // namespace NocML

#endif // NOC_ML_SCALARS_H
