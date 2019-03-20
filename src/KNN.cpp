#include "KNN.h"

KNN::KNN(Rand &r):  SupervisedLearner(), m_rand(r)
{
    dataPoints = make_unique<vector<DataPoint>>();
    featureTypes = make_unique<vector<size_t>>();
}

KNN::~KNN(){}

double KNN::measureDistance(vector<double>& trainingData, const vector<double>& predictingData)
{
    if(trainingData.size() != predictingData.size())
    {
        ThrowError("Attempted to measure distance on two data points with different numbers of features\n");
    }

    //Check environment variable for alterante distance measurement keyword. If none, or eucliedain use euclidean.
    char* measureDistanceEnv = getenv(MEASUREDISTANCEENVVAR);

    if(!measureDistanceEnv || !strcmp(measureDistanceEnv, EUCLIDEANENV))
    {
        return euclideanDistance(trainingData, predictingData);
    }
    else if(!strcmp(measureDistanceEnv, VARIABLEENV))//handle continuous, nominal, and unknown
    {
        return dynamicDistance(trainingData, predictingData);
    }
    else
    {
       ThrowError("Unknown distance measurement method specified\n");
       //This will never be reached but is necessary to avoid a compiler warning
       return -1;
    }
}

double KNN::euclideanDistance(vector<double>& trainingData,const vector<double>& predictingData)
{
    double totalDistance = 0;
    //Sum of all distances, each squared
    for(size_t coordinateIndex = 0; coordinateIndex < trainingData.size(); coordinateIndex++)
    {
        totalDistance += pow((trainingData.at(coordinateIndex) - predictingData.at(coordinateIndex)), 2);
    }

    return sqrt(totalDistance);
}

double KNN::dynamicDistance(vector<double>& trainingData, const vector<double>& predictingData)
{
    double totalDistance = 0;

    for(size_t coordinateIndex = 0; coordinateIndex < trainingData.size(); coordinateIndex++)
    {
        if(trainingData.at(coordinateIndex) == UNKNOWN_VALUE || predictingData.at(coordinateIndex) == UNKNOWN_VALUE)
        {
            totalDistance += 1;
        }
        else if(featureTypes->at(coordinateIndex) == 0)
        {
            vector<double> trainingDataPoint = vector<double>();
            trainingDataPoint.push_back(trainingData.at(coordinateIndex));
            vector<double> predictingDataPoint = vector<double>();
            predictingDataPoint.push_back(predictingData.at(coordinateIndex));
            totalDistance += euclideanDistance(trainingDataPoint, predictingDataPoint);
        }
        else
        {
            totalDistance += !(trainingData.at(coordinateIndex) == predictingData.at(coordinateIndex));
        }
    }
    return totalDistance;
}

Vote KNN::determineVote(DistanceQueueEntry& queueEntry)
{
    //Check environment variable for alterante distance measurement keyword. If none, or eucliedain use euclidean.
    char* weightMethod = getenv(MEASUREWEIGHTENVVAR);

    if(!weightMethod || !strcmp(weightMethod, NOWEIGHTENV))
    {
        return {queueEntry.Data->label, 1};
    }
    else if(!strcmp(weightMethod, DEFAULTWEIGHTENV))
    {
        double distance = queueEntry.distance ? queueEntry.distance : 1.0e-150;//numeric_limits<double>::min();
        return {queueEntry.Data->label , 1/pow(distance,2)};
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
    featureTypes->reserve(featuresCount);

    //determine feature types in case they are needed for distance metrics...
    for(size_t feature = 0; feature < features.cols(); feature++)
    {
        featureTypes->push_back(features.valueCount(feature));
    }

    char* thinData = getenv(THINDATAENVVAR);

    if(!thinData || !strcmp(thinData, NOTHINDATA))
    {
        for(size_t dataIndex = 0; dataIndex < dataPointsCount; dataIndex++)
        {
            dataPoints->push_back({features.row(dataIndex), labels.row(dataIndex).at(0)});
        }
    }
    else if(!strcmp(thinData, THINDATA))
    {
        /*char* minDistance = getenv(MINDISTANCEENV);

        if(!minDistance)
        {
            cerr << "Warning: Environment variable " << MINDISTANCEENV << " is not defined. Setting to 0" << endl;
            minDistance = "0";
        }

        stringstream convertor = stringstream(minDistance);*/

        double minDistanceDouble = 0;

        /*convertor >> minDistanceDouble;*/

        minDistanceDouble = MINDISTANCE;//See note at const definition.

        ofstream logFile("accuracyResults.csv", ios_base::app);

        logFile << minDistanceDouble << ",";

        logFile.close();

        bool tooClose = false;

        for(size_t dataIndex = 0; dataIndex < dataPointsCount; dataIndex++)
        {
            for(DataPoint& comparisonPoint : *dataPoints)
            {
                //Point is too close to be useful so no point in keeping it
                if(measureDistance(comparisonPoint.features, features.row(dataIndex)) < minDistanceDouble)
                {
                    tooClose = true;
                    break;
                }
            }
            
            if(!tooClose)
            {
                dataPoints->push_back({features.row(dataIndex), labels.row(dataIndex).at(0)});
            }
            
            tooClose = false;
        }
    }
    else
    {
        ThrowError("Unknown value for environment variable:", THINDATAENVVAR, "Acceptable values are:", NOTHINDATA, THINDATA);
    }
    
    size_t K = dataPoints->size();

    char* KCount = getenv(KVALUEENVVAR);

    if(KCount && strcmp(KCount, "0"))
    {
        stringstream convertor = stringstream(KCount);

        convertor >> K;
    }

    ofstream logFile("accuracyResults.csv", ios_base::app);

    logFile << dataPoints->size() << "," << K << ",";

    logFile.close();

    cout << "Training finished" << endl;
}

void KNN::predict(const vector<double> &features, vector<double> &labels)
{
    //Create a vector with k spaces (0 < k <= total training points), and go over every point in the training data.
    size_t K = dataPoints->size();

    char* KCount = getenv(KVALUEENVVAR);

    if(KCount && strcmp(KCount, "0"))
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

    char* useRegression = getenv(REGRESSIONENVVAR);

    if(!useRegression || !strcmp(useRegression, NOREGRESSION))
    {
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
                cout << vote.weight << endl;
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
    else if(!strcmp(useRegression, USEREGRESSION))
    {
        //find mean of K neighbors
        double totalValue = 0;
        double totalWeight = 0;
        for(DistanceQueueEntry& neighbor : *nearestNeighbors)
        {
            Vote vote = determineVote(neighbor);
            totalValue += vote.label*vote.weight;
            totalWeight += vote.weight;
        }

        labels.at(0) = totalValue/totalWeight;
    }
    else
    {
        ThrowError("Unknown value entered for regression environment variable.", "Acceptable values are:", USEREGRESSION, NOREGRESSION);
    }
}