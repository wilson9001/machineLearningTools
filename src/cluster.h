#ifndef CLUSTER_H
#define CLUSTER_H

#include <vector>
#include <math.h>
#include <map>
#include "error.h"

using namespace std;

//This struct is used primarily to represent point data outside of clusters
typedef struct point
{
    vector<double>* coordinates;
    size_t id;

    point(size_t id, vector<double>* coordinates): id(id), coordinates(coordinates)
    {}
};

class Cluster
{
    private:
    map<size_t, vector<double>*> mappedPoints;
    vector<double> centroid;
    void reComputeCentroid();
    const string SINGLELINK = "single";
    const string COMPLETELINK = "complete";

    public:
    Cluster();
    Cluster(vector<point> points, bool recomputeCentroid);
    Cluster(vector<double> centroid);
    double calculateSSE();
    void addPoints(vector<point>& pointsToAdd, bool recomputeCentroid);
    double measurePointDistance(vector<double>* otherPoint);
    double measureClusterDistance(Cluster& otherCluster, string measureType);
    vector<double> getCentroid();
    vector<point> getPoints();
    point getPoint(size_t id);
    void removePoint(size_t id, bool recomputeCentroid);
    size_t getPointCount();
};

#endif