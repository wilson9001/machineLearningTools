#include "cluster.h"

Cluster::Cluster()
{
    points = vector<point*>();
    centroid = point();
}

Cluster::Cluster(vector<point*> points): points(points)
{
    centroid = point();
}

double Cluster::calculateSSE()
{
    double SSE = 0;

    for(point* point : points)
    {
        SSE += pow(measurePointDistance(point), 2);
    }

    return SSE;
}

void Cluster::addPoints(vector<point*> pointsToAdd)
{
    points.reserve(points.size() + pointsToAdd.size());

    points.insert(points.end(), pointsToAdd.begin(), pointsToAdd.end());
}

double Cluster::measurePointDistance(point* otherPoint)
{
    if(centroid.size() != otherPoint->size())
    {
        ThrowError("Points to compare do not have the same dimensionality:", to_string(centroid.size()), to_string(otherPoint->size()));
    }

    recomputeCentroid();

    double totalDistance = 0;

    for(size_t coordinate = 0; coordinate < centroid.size(); coordinate++)
    {
        totalDistance += pow((centroid.at(coordinate) - otherPoint->at(coordinate)), 2);
    }

    return sqrt(totalDistance);
}

void Cluster::recomputeCentroid()
{
    centroid.clear();
    centroid.reserve(points.at(0)->size());
    double currentDimensionLength;

    for(size_t dimension = 0; dimension < centroid.size(); dimension++)
    {
        currentDimensionLength = 0;

        for(size_t pointIndex = 0; pointIndex < points.size(); pointIndex++)
        {
            currentDimensionLength += points.at(pointIndex)->at(dimension);
        }

        centroid.push_back(currentDimensionLength/points.size());
    }
}

point Cluster::getCentroid()
{
    recomputeCentroid();
    return centroid;
}

vector<point*> Cluster::getPoints()
{
    return points;
}