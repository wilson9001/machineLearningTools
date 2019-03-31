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

    vector<point*> pointForNewCluster = vector<point*>();

    if(initialPoint == RANDOMINITCENTROIDS)
    {
        shuffle(points.begin(), points.end(), default_random_engine());
    }

    size_t pointIndex;

    for (pointIndex = 0; pointIndex < k; pointIndex++)
    {
        pointForNewCluster.clear();        
        pointForNewCluster.push_back(points.at(pointIndex));
        clusters.push_back(make_unique<Cluster>(pointForNewCluster));
    }

    //TODO: Add remaining points from index into clusters by measuring against the centroid of each cluster

    //TODO: Set a change occurred flag to false and Loop through each cluster to check if each point is closer to a newly computed centroid.
    //If it is then remove it from the cluster and place it in the now closer centroid and set the changed flag to true.
    //Repeat until the change occurred flag stays false. This means that no points moved and they are therefore optimally clustered.  
}

void Graph::HAC(string linkType, size_t lowerK, size_t upperK)
{
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

void Graph::train(Matrix &features, Matrix &labels)
{
    char *useLabel = getenv(INCLUDELABELENV);

    if (useLabel && !strcmp(useLabel, YESSTR))
    {
        for (size_t dataPointIndex = 0; dataPointIndex < features.rows(); dataPointIndex++)
        {
            point &dataPoint = features.row(dataPointIndex);
            dataPoint.push_back(labels.row(dataPointIndex).at(0));
        }
    }

    points.reserve(features.rows());

    for (size_t row = 0; row < features.rows(); row++)
    {
        points.push_back(&features.row(row));
    }

    createClusters();

    ofstream outFile("accuracyResults.csv", ofstream::app);

    outFile << getenv(CLUSTERALGOENV) << ":" << clusters.size() << endl;

    point centroid;

    for (unique_ptr<Cluster> &cluster : clusters)
    {
        centroid = cluster->getCentroid();

        for (double coordinate : centroid)
        {
            outFile << coordinate << ",";
        }

        outFile << ";" << cluster->getPoints().size() << cluster->calculateSSE() << endl;
    }

    outFile << calculateTotalSSE() << endl
            << endl;

    outFile.close();
}

void Graph::predict(const vector<double> &features, vector<double> &labels)
{
    //Don't think I use this...
}