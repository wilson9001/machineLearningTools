#include "cluster.h"

Cluster::Cluster()
{
    mappedPoints = map<size_t, vector<double>*>();
    centroid = vector<double>();
}

Cluster::Cluster(vector<point> points, bool recomputeCentroid)
{
    mappedPoints = map<size_t, vector<double>*>();

    for (point currentPoint : points)
    {
        mappedPoints.emplace(currentPoint.id, currentPoint.coordinates);
    }

    centroid = vector<double>();
    
    if(recomputeCentroid)
    {
        reComputeCentroid();
    }
}

Cluster::Cluster(vector<double> centroid): centroid(centroid)
{
    mappedPoints = map<size_t, vector<double>*>();
}

double Cluster::calculateSSE()
{
    double SSE = 0;

    for(map<size_t, vector<double>*>::iterator mapIter = mappedPoints.begin(); mapIter != mappedPoints.end(); mapIter++)
    {
        SSE += pow(measurePointDistance(mapIter->second), 2);
    }

    return SSE;
}

void Cluster::addPoints(vector<point>& pointsToAdd, bool recomputeCentroid)
{
    if(!pointsToAdd.empty())
    {
        for(point& pointToAdd : pointsToAdd)
        {
            mappedPoints.emplace(pointToAdd.id, pointToAdd.coordinates);
        }
    }

    if(recomputeCentroid)
    {
        reComputeCentroid();
    }
    
}

double Cluster::measurePointDistance(vector<double>* otherPoint)
{
    if(centroid.size() != otherPoint->size())
    {
        ThrowError("Points to compare do not have the same dimensionality:", to_string(centroid.size()), to_string(otherPoint->size()));
    }

    double totalDistance = 0;

    for(size_t coordinate = 0; coordinate < centroid.size(); coordinate++)
    {
        totalDistance += pow((centroid.at(coordinate) - otherPoint->at(coordinate)), 2);
    }

    return sqrt(totalDistance);
}

void Cluster::reComputeCentroid()
{
    centroid.clear();
    centroid.reserve(mappedPoints.begin()->second->size());
    double currentDimensionLength;

    for(size_t dimension = 0; dimension < mappedPoints.begin()->second->size(); dimension++)
    {
        currentDimensionLength = 0;

        for(map<size_t, vector<double>*>::iterator mapIter = mappedPoints.begin(); mapIter != mappedPoints.end(); mapIter++)
        {
            currentDimensionLength += mapIter->second->at(dimension);
        }

        centroid.push_back(currentDimensionLength/mappedPoints.size());
    }
}

vector<double> Cluster::getCentroid()
{
    return centroid;
}

vector<point> Cluster::getPoints()
{
    vector<point> points = vector<point>();

    points.reserve(mappedPoints.size());

    for(map<size_t, vector<double>*>::iterator mapIter = mappedPoints.begin(); mapIter != mappedPoints.end(); mapIter++)
    {
        points.push_back(point(mapIter->first, mapIter->second));
    }

    return points;
}

point Cluster::getPoint(size_t id)
{
    return point(id, mappedPoints.at(id));
}

void Cluster::removePoint(size_t id, bool recomputeCentroid)
{
    mappedPoints.erase(id);

    if(recomputeCentroid)
    {
        reComputeCentroid();
    }
}

size_t Cluster::getPointCount()
{
    return mappedPoints.size();
}