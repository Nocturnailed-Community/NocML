#ifndef NOC_ML_KMEANS_H
#define NOC_ML_KMEANS_H

#include "../Utils/MathUtils.h"

namespace NocML {

    class KMeans {
    public:
        /**
         * @param k Number of clusters
         * @param dims Number of features
         * @param centroids Pointer to storage for centroids (size: k * dims)
         */
        KMeans(size_t k, size_t dims, float* centroids)
            : _k(k), _dims(dims), _centroids(centroids) {}

        /**
         * Run the K-Means algorithm on a dataset.
         * @param data Pointer to input data (size: num_samples * dims)
         * @param num_samples Number of samples in dataset
         * @param assignments Pointer to store resulting cluster IDs for each sample (size: num_samples)
         * @param max_iterations Maximum number of iterations
         */
        void run(const float* data, size_t num_samples, int* assignments, int max_iterations = 10);

        /**
         * Predict the nearest cluster for a single data point.
         */
        int predict(const float* input);

        /**
         * Manually set a centroid (e.g. for initialization).
         */
        void setCentroid(size_t clusterIdx, const float* values) {
            if (clusterIdx >= _k) return;
            for (size_t i = 0; i < _dims; i++) {
                _centroids[clusterIdx * _dims + i] = values[i];
            }
        }

    private:
        size_t _k;
        size_t _dims;
        float* _centroids;
    };

} // namespace NocML

#endif // NOC_ML_KMEANS_H
