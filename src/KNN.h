#ifndef KNN_H
#define KNN_H

#include <vector>
#include <memory>
#include <string.h>
#include <iostream>
#include <math.h>
#include <algorithm>
#include <fstream>
#include "error.h"
#include "learner.h"

using namespace std;

struct DataPoint
{
    vector<double> features;
    double label;
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

class KNN : public SupervisedLearner
{
    private:
    Rand &m_rand;
    unique_ptr<vector<DataPoint>> dataPoints;
    unique_ptr<vector<size_t>> featureTypes;
    double measureDistance(vector<double>& trainingData, const vector<double>& predictingData);
    Vote determineVote(DistanceQueueEntry& queueEntry);
    double euclideanDistance(vector<double>& trainingData, const vector<double>& predictingData);
    double dynamicDistance(vector<double>& trainingData, const vector<double>& predictingData);
    const double MINDISTANCE = 0.001; //This will need to be used until I can figure out why the environment variable doesn't change when read in even though it appears to change in BASH...
    //add more distance measuring functions later if desired.

    public:
    const char* EUCLIDEANENV = "EUCLIDEAN";
    const char* VARIABLEENV = "DYNAMIC";
    const char* MEASUREDISTANCEENVVAR = "distance";
    const char* KVALUEENVVAR = "KValue";
    const char* MEASUREWEIGHTENVVAR = "weight";
    const char* DEFAULTWEIGHTENV = "inverseD";
    const char* NOWEIGHTENV = "none";
    const char* REGRESSIONENVVAR = "regression";
    const char* USEREGRESSION = "y";
    const char* NOREGRESSION = "n";
    const char* THINDATAENVVAR = "thinData";
    const char* THINDATA = "y";
    const char* NOTHINDATA = "n";
    const char* MINDISTANCEENV = "minDistance";
    //add more environment strings to use future measuring functions later if desired
    
    KNN(Rand &r);
    ~KNN();

    // Train the model to predict the labels
    void train(Matrix &features, Matrix &labels);
    
    // Evaluate the features and predict the labels
    void predict(const vector<double> &features, vector<double> &labels);
};

#endif