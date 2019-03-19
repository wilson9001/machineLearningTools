#include "KNN.h"

KNN::KNN(Rand &r):  SupervisedLearner(), m_rand(r)
{
    dataPoints = make_unique<vector<DataPoint>>();
    medianValues = make_unique<vector<double>>();
}

KNN::~KNN(){}

double KNN::measureDistance(vector<double>* trainingData, const vector<double>& predictingData)
{
    if(trainingData->size() != predictingData.size())
    {
        ThrowError("Attempted to measure distance on two data points with different numbers of features\n");
    }

    //Check environment variable for alterante distance measurement keyword. If none, or eucliedain use euclidean.
    char* measureDistanceEnv = getenv(MEASUREDISTANCEENVVAR);

    if(!measureDistanceEnv || !strcmp(measureDistanceEnv, EUCLIDEANENV))
    {
        return euclideanDistance(trainingData, predictingData);
    }
    else
    {
       ThrowError("Unknown distance measurement method specified\n");
       //This will never be reached but is necessary to avoid a compiler error
       return -1;
    }
}

double KNN::euclideanDistance(vector<double>* trainingData,const vector<double>& predictingData)
{
    double totalDistance = 0;
    //Sum of all distances, each squared
    for(size_t coordinateIndex = 0; coordinateIndex < trainingData->size(); coordinateIndex++)
    {
        totalDistance += pow((trainingData->at(coordinateIndex) - predictingData.at(coordinateIndex)), 2);
    }

    return sqrt(totalDistance);
}

//TODO: Implement this
Vote KNN::determineVote(DistanceQueueEntry& queueEntry)
{
    //Check environment variable for alterante distance measurement keyword. If none, or eucliedain use euclidean.
    char* weightMethod = getenv(MEASUREWEIGHTENVVAR);

    if(!weightMethod || !strcmp(weightMethod, DEFAULTWEIGHTENV))
    {
        return {*queueEntry.Data->label , 1/queueEntry.distance};
    }
    else
    {
       ThrowError("Unknown weight measurement method specified\n");
       //This will never be reached but is necessary to avoid a compiler error
       return {-1, -1};
    }
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

    char* KCount = getenv(KVALUEENVVAR);

    if(KCount && !strcmp(KCount, "0"))
    {
        stringstream convertor = stringstream(KCount);

        convertor >> K;
    }

    unique_ptr<vector<DistanceQueueEntry>> nearestNeighbors = make_unique<vector<DistanceQueueEntry>>();

    nearestNeighbors->reserve(K+1);

    double distance;
    size_t oldSize;

    //Maintain the distance ordering of the vector by iterating over it and comparing diatances in the structs
    for(size_t dataPointIndex = 0; dataPointIndex < dataPoints->size(); dataPointIndex++)
    {
        distance = measureDistance(dataPoints->at(dataPointIndex).features, features);

        oldSize = nearestNeighbors->size();
        for(size_t queueIndex = 0; queueIndex < nearestNeighbors->size(); queueIndex++)
        {
            //if the data point is closer than other points in the vector, insert it at the appropriate position
            if(nearestNeighbors->at(queueIndex).distance > distance)
            {
                nearestNeighbors->insert(nearestNeighbors->begin()+queueIndex, {&dataPoints->at(dataPointIndex), distance});
                break;
            }
        }

        //Add point to end of stack if it wasn't already added and the vector is still filling up to size K
        if(nearestNeighbors->size() == oldSize && nearestNeighbors->size() < K)
        {
            nearestNeighbors->push_back({&dataPoints->at(dataPointIndex), distance});
        }
        //Pop the end off the vector if its size exceeds the limit of nodes voting
        //This will be the currently farthest away node in the vector
        else if(nearestNeighbors->size() > K)
        {
            nearestNeighbors->pop_back();
        }
    }
    
    //For each entry still in the queue, call the obtain vote function to get the label votes and map the labels to their aggregate strengths
    unique_ptr<map<double, double>> labelToVoteStrength = make_unique<map<double, double>>();

    for(DistanceQueueEntry& queueEntry : *nearestNeighbors)
    {
        Vote vote = determineVote(queueEntry);

        if(!labelToVoteStrength->count(vote.label))
        {
            labelToVoteStrength->emplace(vote.label, vote.weight);
        }
        else
        {
            labelToVoteStrength->at(vote.label) += vote.weight;
        }
    }

    //find strongest vote
    double finalLabel = -1;
    double currentGreatestWeight = -1;

    for(map<double, double>::iterator voteIterator = labelToVoteStrength->begin(); voteIterator != labelToVoteStrength->end(); voteIterator++)
    {
        if(voteIterator->second > currentGreatestWeight)
        {
            currentGreatestWeight = voteIterator->second;
            finalLabel = voteIterator->first;
        }
    }

    labels.at(0) = finalLabel;
}