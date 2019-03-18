#ifndef KNN_H
#define KNN_H

#include <vector>
#include <memory>
#include <string.h>
#include <iostream>
#include <math.h>
#include <algorithm>
#include "error.h"
#include "learner.h"

using namespace std;

class KNN : public SupervisedLearner
{
    private:
    struct DataPoint
    {
        vector<double>* features;
        double* label;
    };

    struct DistanceQueueEntry
    {
        DataPoint* Data;
        double distance;
    };

    struct Vote
    {
        double label;
        double weight;
    };

    Rand &m_rand;
    unique_ptr<vector<DataPoint>> dataPoints;
    unique_ptr<vector<double>> medianValues;
    double measureDistance(vector<double>& coordinates1, vector<double>& coordinates2);
    Vote determineVote(DistanceQueueEntry& queueEntry);
    double euclideanDistance(vector<double>& coordinates1, vector<double>& coordinates2);
    //add more distance measuring functions later if desired.

    public:
    const char* EUCLIDEANENV = "EUCLIDEAN";
    const char* MEASUREDISTANCEENVVAR = "distance";
    const char* KVALUE = "KValue";

    //add more environment strings to use future measuring functions later if desired
    
    KNN(Rand &r);
    ~KNN();

    // Train the model to predict the labels
    void train(Matrix &features, Matrix &labels);
    
    // Evaluate the features and predict the labels
    void predict(const vector<double> &features, vector<double> &labels);
};

#endif