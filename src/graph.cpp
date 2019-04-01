#include "graph.h"

Graph::Graph(Rand &r) : SupervisedLearner(), m_rand(r)
{
    clusters = vector<unique_ptr<Cluster>>();
    points = vector<point *>();
}

Graph::~Graph()
{
}

void Graph::KMeans(size_t k, string initialPoint)
{
    if (k > points.size())
    {
        cerr << "WARNING: k value " << k << "is greater than total data points " << points.size() << "\nK will be reduced so each point will be its own cluster\n";
        k = points.size();
    }

    if (k == 0)
    {
        cerr << "WARNING: k == 0. You must have at least one cluster, so k will be changed to 1\n";
        k = 1;
    }

    if(initialPoint == RANDOMINITCENTROIDS)
    {
        cout << "NOTE: Points were shuffled in order to use random points for initial centroids" << endl;

        shuffle(points.begin(), points.end(), default_random_engine());
    }

    size_t pointIndex;

    //create initial clusters
    for (pointIndex = 0; pointIndex < k; pointIndex++)
    {
        clusters.push_back(make_unique<Cluster>(*(points.at(pointIndex).coordinates)));
    }

    vector<vector<point>> vectorsToAddToClusters = vector<vector<point>>();

    vectorsToAddToClusters.reserve(k);

    for(size_t i = 0; i < clusters.size(); i++)
    {
        vectorsToAddToClusters.push_back(vector<point>());
    }

    size_t clusterNumber;

    //place points in clusters initially
    for(pointIndex; pointIndex < points.size(); pointIndex++)
    {
        clusterNumber = findClosestCentroid(points.at(pointIndex));
        vectorsToAddToClusters.at(clusterNumber).push_back(points.at(pointIndex));
    }

    for(size_t clusterIndex = 0; clusterIndex < clusters.size(); clusterIndex++)
    {
        clusters.at(clusterIndex)->addPoints(vectorsToAddToClusters.at(clusterIndex), true);
    }

    //Move points around and recalculate centroids until no more points move around.
    bool pointsChanged;
    vector<point> pointsToConsider = vector<point>();

    do
    {
        pointsChanged = false;

        //Clean record of what points go where
        for(vector<point>& clusterVector : vectorsToAddToClusters)
        {
            clusterVector.clear();
        }

        //for each centroid
        for(size_t clusterIndex = 0; clusterIndex < clusters.size(); clusterIndex++)
        {
            pointsToConsider = clusters.at(clusterIndex)->getPoints();

            //for each point in each centroid
            for(point& currentPoint : pointsToConsider)
            {
                size_t closestCentroidIndex = findClosestCentroid(currentPoint);

                //track points which are closer to new centroids
                if(closestCentroidIndex != clusterIndex)
                {
                    vectorsToAddToClusters.at(closestCentroidIndex).push_back(currentPoint);
                    clusters.at(clusterIndex)->removePoint(currentPoint.id, false);
                    pointsChanged = true;
                }
            }
        }

        // actually move points into new clusters and recalculate centroids
        for(size_t clusterIndex = 0; clusterIndex < vectorsToAddToClusters.size(); clusterIndex++)
        {
            clusters.at(clusterIndex)->addPoints(vectorsToAddToClusters.at(clusterIndex), true);
        }

    } while(pointsChanged);
}

void Graph::HAC(string linkType, size_t lowerK, size_t upperK)
{
    ThrowError("Sorry, HAC is not yet implemented");
}

void Graph::createClusters()
{
    char *clusteringAlgo = getenv(CLUSTERALGOENV);

    if (!clusteringAlgo)
    {
        ThrowError("Clustering algorithm must be defined via env variable:", CLUSTERALGOENV);
    }

    if (!strcmp(clusteringAlgo, KMEANSENV))
    {
        char *kStr = getenv(KMEANSENV);

        if (!kStr)
        {
            ThrowError("k must be defined via env variable:", KMEANSENV);
        }

        char *initialCentroids = getenv(INITCENTROIDSENV);

        if (!initialCentroids)
        {
            ThrowError("Initial centroids must be defined via env variable:", INITCENTROIDSENV);
        }

        stringstream convertor(kStr);

        size_t k;

        convertor >> k;

        KMeans(k, static_cast<string>(initialCentroids));
    }
    else if (!strcmp(clusteringAlgo, HACSTR))
    {
        char *linkType = getenv(LINKTYPEENV);

        if (!linkType)
        {
            ThrowError("Link type must be defined on env var:", LINKTYPEENV);
        }

        char *lowerKStr = getenv(KLOWERENV);
        char *upperKStr = getenv(KUPPERENV);

        if (!lowerKStr || !upperKStr)
        {
            ThrowError("Both upper and lower k-values must be defined via env vars:", KLOWERENV, KUPPERENV);
        }

        stringstream convertor(lowerKStr);

        size_t lowerK, upperK;

        convertor >> lowerK;

        convertor << upperKStr;

        convertor >> upperK;

        HAC(static_cast<string>(linkType), lowerK, upperK);
    }
    else
    {
        ThrowError("Unknown algorithm specified:", clusteringAlgo);
    }
}

double Graph::calculateTotalSSE()
{
    double totalSSE = 0;

    for (unique_ptr<Cluster> &cluster : clusters)
    {
        totalSSE += cluster->calculateSSE();
    }

    return totalSSE;
}

size_t Graph::findClosestCentroid(point pointToEvaluate)
{
    double currentDistance;
    double closestDistance = numeric_limits<double>::max();
    size_t clusterNumber = numeric_limits<size_t>::max();

        for(size_t clusterIndex = 0; clusterIndex < clusters.size(); clusterIndex++)
        {
            currentDistance = clusters.at(clusterIndex)->measurePointDistance(pointToEvaluate.coordinates);

            if(currentDistance < closestDistance)
            {
                closestDistance = currentDistance;
                clusterNumber = clusterIndex;
            }
        }

    return clusterNumber;
}

void Graph::train(Matrix &features, Matrix &labels)
{
    char *useLabel = getenv(INCLUDELABELENV);

    if (useLabel && !strcmp(useLabel, YESSTR))
    {
        for (size_t dataPointIndex = 0; dataPointIndex < features.rows(); dataPointIndex++)
        {
            features.row(dataPointIndex).push_back(labels.row(dataPointIndex).at(0));
        }
    }

    points.reserve(features.rows());

    for (size_t row = 0; row < features.rows(); row++)
    {
        points.push_back(point(row, &features.row(row)));
    }

    createClusters();

    ofstream outFile("accuracyResults.csv", ofstream::app);

    outFile << getenv(CLUSTERALGOENV) << ":" << clusters.size() << endl;

    vector<double> centroid;

    for (unique_ptr<Cluster> &cluster : clusters)
    {
        centroid = cluster->getCentroid();

        for (double coordinate : centroid)
        {
            outFile << coordinate << ",";
        }

        outFile << ";" << cluster->getPointCount() << cluster->calculateSSE() << endl;
    }

    outFile << calculateTotalSSE() << endl << endl;

    outFile.close();
}

void Graph::predict(const vector<double> &features, vector<double> &labels)
{
    //Don't think I use this...
}