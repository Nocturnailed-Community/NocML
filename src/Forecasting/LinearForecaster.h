#ifndef NOC_ML_LINEAR_FORECASTER_H
#define NOC_ML_LINEAR_FORECASTER_H

#include <Arduino.h>

namespace NocML {

    class LinearForecaster {
    public:
        LinearForecaster() : _slope(0), _intercept(0), _n(0) {}

        /**
         * Fit a simple linear regression line (y = mx + b)
         * @param x Times or indices (independent variable)
         * @param y Values (dependent variable)
         * @param n Number of data points
         */
        void fit(const float* x, const float* y, size_t n) {
            if (n < 2) return;
            _n = n;

            float xSum = 0, ySum = 0, xySum = 0, xxSum = 0;
            for (size_t i = 0; i < n; i++) {
                xSum += x[i];
                ySum += y[i];
                xySum += x[i] * y[i];
                xxSum += x[i] * x[i];
            }

            float denominator = (n * xxSum - xSum * xSum);
            if (abs(denominator) < 1e-6) return;

            _slope = (n * xySum - xSum * ySum) / denominator;
            _intercept = (ySum - _slope * xSum) / n;
        }

        /**
         * Predict the value for a given x.
         */
        float predict(float x) {
            return _slope * x + _intercept;
        }

        /**
         * Forecast the next value given the last x-index.
         */
        float forecastNext() {
            return predict((float)_n);
        }

        float getSlope() { return _slope; }
        float getIntercept() { return _intercept; }

    private:
        float _slope;
        float _intercept;
        size_t _n;
    };

} // namespace NocML

#endif // NOC_ML_LINEAR_FORECASTER_H
