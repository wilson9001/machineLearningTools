#ifndef GRAPH_H
#define GRAPH_H

#include "learner.h"
#include "cluster.h"
#include "matrix.h"
#include <memory>
#include <string.h>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <random>

class Graph: SupervisedLearner {
    private:
    vector<unique_ptr<Cluster>> clusters;
    vector<point> points;
    Rand m_rand;
    const char* CLUSTERALGOENV = "algorithm";
    const char* HACSTR = "HAC";
    const char* KMEANSENV = "kMeans";
    const char* INITCENTROIDSENV = "initPoints";
    const char* RANDOMINITCENTROIDS = "random";
    const char* STARTINITCENTROIDS = "start";
    const char* LINKTYPEENV = "linkType";
    const char* INCLUDELABELENV = "includeLabel";
    const char* YESSTR = "yes";
    const char* NOSTR = "no";
    const char* KLOWERENV = "lowerK";
    const char* KUPPERENV = "higherK";

    public:
    Graph(Rand &r);
    ~Graph();
    void createClusters();
    void KMeans(size_t k, string initialPoints);
    void HAC(string linkType, size_t lowerK, size_t upperK);

    double calculateTotalSSE();
    size_t findClosestCentroid(point pointToEvaluate);

    // Train the model to predict the labels
    void train(Matrix &features, Matrix &labels);
    
    // Evaluate the features and predict the labels
    void predict(const vector<double> &features, vector<double> &labels);
};

#endif