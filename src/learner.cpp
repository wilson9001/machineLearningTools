// ----------------------------------------------------------------
// The contents of this file are distributed under the CC0 license.
// See http://creativecommons.org/publicdomain/zero/1.0/
// ----------------------------------------------------------------

#include "learner.h"
#include "error.h"
#include <iostream>
#include <fstream>
#include <map>
#include <vector>
#include <algorithm>
#include <sstream>
#include <math.h>
#include <iostream>

using std::vector;
using std::cout;

double SupervisedLearner::measureAccuracy(Matrix& features, Matrix& labels, Matrix* pOutStats)
{
	// Check assumptions
	if(features.rows() != labels.rows())
		ThrowError("Expected the features and labels to have the same number of rows");
	if(labels.cols() != 1)
		ThrowError("Sorry, this method currently only supports one-dimensional labels");
	if(features.rows() == 0)
		ThrowError("Expected at least one row");

	// Measure Accuracy
	size_t labelValues = labels.valueCount(0);
	if(labelValues == 0) // If the label is continuous...
	{
		// The label is continuous, so measure root mean squared error
		vector<double> pred;
		pred.resize(1);
		double sse = 0.0;
		for(size_t i = 0; i < features.rows(); i++)
		{
			const vector<double>& feat = features.row(i);
			vector<double>& targ = labels.row(i);
			pred[0] = 0.0; // make sure the prediction is not biassed by a previous prediction
			predict(feat, pred);
			double delta = targ[0] - pred[0];
			sse += (delta * delta);
		}
		return sqrt(sse / features.rows());
	}
	else
	{
		// The label is nominal, so measure predictive accuracy
		if(pOutStats)
		{
			pOutStats->setSize(2, labelValues);
			pOutStats->setAll(0.0);
		}
		size_t correctCount = 0;
		vector<double> prediction;
		prediction.resize(1);
		for(size_t i = 0; i < features.rows(); i++)
		{
			const vector<double>& feat = features.row(i);
			size_t targ = (size_t)labels[i][0];
			if(targ >= labelValues)
				ThrowError("The label is out of range");
			if(pOutStats)
				(*pOutStats)[1][targ]++; // increment the "possible" count
			predict(feat, prediction);
			size_t pred = (size_t)prediction[0];
			if(pred == targ)
			{
				correctCount++;
				if(pOutStats)
					(*pOutStats)[0][targ]++; // increment the "correct" count
			}
		}
		return (double)correctCount / features.rows();
	}
}

double SupervisedLearner::crossValidate(size_t reps, size_t folds, Matrix& features, Matrix& labels, Rand& r, bool verbose)
{
	if(features.rows() != labels.rows())
		ThrowError("Expected the features and labels to have the same number of rows");
	double sum = 0.0;
	for(size_t rep = 0; rep < reps; rep++)
	{
		features.shuffleRows(r, &labels);
		size_t foldSize = features.rows() / folds;
		for(size_t fold = 0; fold < folds; fold++)
		{
			// Make the training and test sets
			size_t foldBegin = foldSize * fold;
			size_t foldEnd = foldBegin + foldSize;
			Matrix trainFeatures, trainLabels, testFeatures, testLabels;
			trainFeatures.copyPart(features, 0, 0, foldBegin, features.cols());
			trainFeatures.copyPart(features, foldEnd, 0, features.rows() - foldEnd, features.cols());
			trainLabels.copyPart(labels, 0, 0, foldBegin, labels.cols());
			trainLabels.copyPart(labels, foldEnd, 0, labels.rows() - foldEnd, labels.cols());
			testFeatures.copyPart(features, foldBegin, 0, foldEnd - foldBegin, features.cols());
			testLabels.copyPart(labels, foldBegin, 0, foldEnd - foldBegin, labels.cols());

			// Train
			train(trainFeatures, trainLabels);

			// Test
			double accuracy = measureAccuracy(testFeatures, testLabels);
			sum += accuracy;

			// Print intermediate results
			if(verbose)
			{
				cout << "Rep: " << rep << ", Fold: " << fold << ", Accuracy: " << accuracy << "\n";
				cout.flush();
			}
		}
	}
	return sum / (reps * folds);
}

