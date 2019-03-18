#include "KNN.h"

KNN::KNN(Rand &r):  SupervisedLearner(), m_rand(r)
{
    dataPoints = make_unique<vector<DataPoint>>();
    medianValues = make_unique<vector<double>>();
}

KNN::~KNN(){}

double KNN::measureDistance(vector<double>& coordinates1, vector<double>& coordinates2)
{
    if(coordinates1.size() != coordinates2.size())
    {
        ThrowError("Attempted to measure distance on two data points with different numbers of features\n");
    }

    //Check environment variable for alterante distance measurement keyword. If none, or eucliedain use euclidean.
    char* measureDistanceEnv = getenv(MEASUREDISTANCEENVVAR);

    if(!measureDistanceEnv || !strcmp(measureDistanceEnv, EUCLIDEANENV))
    {
        return euclideanDistance(coordinates1, coordinates2);
    }
    else
    {
       ThrowError("Unknown distance measurement method specified\n");
       //This will never be reached but is necessary to avoid a compiler error
       return -1;
    }

}

double KNN::euclideanDistance(vector<double>& coordinates1, vector<double>& coordinates2)
{
    double totalDistance = 0;
    //Sum of all distances, each squared
    for(size_t coordinateIndex = 0; coordinateIndex < coordinates1.size(); coordinateIndex++)
    {
        totalDistance += pow((coordinates1.at(coordinateIndex) - coordinates2.at(coordinateIndex)), 2);
    }

    return sqrt(totalDistance);
}

//TODO: Implement this
Vote KNN::determineVote(DistanceQueueEntry& queueEntry)
{

}

void KNN::train(Matrix &features, Matrix& labels)
{
    size_t dataPointsCount = features.rows();
    size_t featuresCount = features.cols();

    //set up instance variables
    dataPoints->reserve(dataPointsCount);
    medianValues->reserve(featuresCount);

    //determine medians
    unique_ptr<vector<vector<double>>> featureValueAggregator = make_unique<vector<vector<double>>>();

    featureValueAggregator->reserve(featuresCount);

    for(size_t i = 0; i < featuresCount; i++)
    {
        featureValueAggregator->push_back(vector<double>());
        featureValueAggregator->back().reserve(dataPointsCount);
    }

    for(size_t dataIndex = 0; dataIndex < dataPointsCount; dataIndex++)
    {
        //store median value of each point to fill in for missing values
        vector<double>& currentFeatureRow = features.row(dataIndex);

        for(size_t featureIndex = 0; featureIndex < currentFeatureRow.size(); featureIndex++)
        {
            //Check that value is not unknown
            if(currentFeatureRow.at(featureIndex) != UNKNOWN_VALUE)
            {
                featureValueAggregator->at(featureIndex).push_back(currentFeatureRow.at(featureIndex));
            }
        }

        dataPoints->push_back({&currentFeatureRow, &labels.row(dataIndex).at(0)});
    }

    for(size_t featureIndex = 0; featureIndex < featuresCount; featureIndex++)
    {
        vector<double>& featureValues = featureValueAggregator->at(featureIndex);

        sort(featureValues.begin(), featureValues.end());

        double medianValue = 0;

        if(!featureValues.empty())
        {
            medianValue = featureValues.at((featureValues.size()-1)/2);
        }

        medianValues->push_back(medianValue);
    }
}

void KNN::predict(const vector<double> &features, vector<double> &labels)
{
    //TODO: GO through features and test for missing values. Fill in any missing values with median value
    vector<double> featuresCopy = vector<double>(features);

    for(size_t featureIndex = 0; featureIndex < featuresCopy.size(); featureIndex++)
    {
        if(featuresCopy.at(featureIndex) == UNKNOWN_VALUE)
        {
            featuresCopy.at(featureIndex) = medianValues->at(featureIndex);
        }
    }

    //TODO: Create a vector with k spaces (0 < k <= total training points), and go over every point in the training data.
    size_t K = dataPoints->size();

    char* KCount = getenv(KVALUE);

    if(KCount && !strcmp(KCount, "0"))
    {
        stringstream convertor = stringstream(KCount);

        convertor >> K;
    }

    unique_ptr<vector<DistanceQueueEntry>> nearestNeighbors = make_unique<vector<DistanceQueueEntry>>();

    nearestNeighbors->reserve(K+1);

    //Maintain the distance ordering of the vector by iterating over it and comparing diatances in the structs
    
    //Pop the end off the vector if its size exceeds the limit of nodes voting.

    //For each entry still in the queue, call the obtain vote function to get the label votes and their strengths. Map them into aggregate scores and set the label as the highest value.
}