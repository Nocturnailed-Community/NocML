#ifndef NOC_ML_H
#define NOC_ML_H

/**
 * NocML - A lightweight Machine Learning library for Arduino.
 * 
 * Supports:
 * - Classification (KNN, Naive Bayes, Logistic Regression, Decision Tree)
 * - Clustering (K-Means)
 * - Forecasting (Linear Regression)
 * - Preprocessing (Scalers, Polynomial Features)
 */

// Core Utilities
#include "Utils/MathUtils.h"

// Preprocessing
#include "Preprocessing/Scalers.h"
#include "Preprocessing/PolynomialFeatures.h"

// Classification
#include "Classification/KNN.h"
#include "Classification/NaiveBayes.h"
#include "Classification/LogisticRegression.h"
#include "Classification/DecisionTree.h"

// Clustering
#include "Clustering/KMeans.h"

// Forecasting
#include "Forecasting/LinearForecaster.h"

#endif // NOC_ML_H
