// ----------------------------------------------------------------
// The contents of this file are distributed under the CC0 license.
// See http://creativecommons.org/publicdomain/zero/1.0/
// ----------------------------------------------------------------

#ifndef LEARNER_H
#define LEARNER_H


#include "matrix.h"
#include "rand.h"
#include <vector>

// This is the base class of supervised learning algorithms.
class SupervisedLearner
{
public:
	SupervisedLearner() {}
	virtual ~SupervisedLearner() {}

	// Train the model to predict the labels
	virtual void train(Matrix& features, Matrix& labels) = 0;

	// Evaluate the features and predict the labels
	virtual void predict(const std::vector<double>& features, std::vector<double>& labels) = 0;

	// The model must be trained before you call this method. If the label is nominal,
	// it returns the predictive accuracy. If the label is continuous, it returns
	// the root mean squared error (RMSE). If pOutStats is non-NULL, and the
	// output label is nominal, then pOutStats will be resized and set to
	// contain stats about the frequency of correct predictions with each label.
	double measureAccuracy(Matrix& features, Matrix& labels, Matrix* pOutStats = NULL);

	// Performs cross-validation. If verbose is true, it prints the accuracy to
	// cout after each fold.
	double crossValidate(size_t reps, size_t folds, Matrix& features, Matrix& labels, Rand& r, bool verbose = false);
};


#endif // LEARNER_H
