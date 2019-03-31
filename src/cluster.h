#ifndef CLUSTER_H
#define CLUSTER_H

#include <vector>
#include <math.h>
#include "error.h"

using namespace std;

typedef vector<double> point;

class Cluster {
    private:
    vector<point*> points;
    point centroid;
    void recomputeCentroid();
    const string SINGLELINK = "single";
    const string COMPLETELINK = "complete";

    public:
    Cluster();
    Cluster(vector<point*> points);
    double calculateSSE();
    void addPoints(vector<point*> pointsToAdd);
    double measurePointDistance(point* otherPoint);
    double measureClusterDistance(Cluster& otherCluster, string measureType);
    point getCentroid();
    vector<point*> getPoints();
};

#endif