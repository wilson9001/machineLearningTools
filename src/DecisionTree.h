#ifndef DECISIONTREE_H
#define DECISIONTREE_H

#include "DTNode.h"
#include "learner.h"
#include "error.h"
#include <memory>
#include <math.h>
#include <iostream>
#include <tuple>
#include <string.h>

class DecisionTree : public SupervisedLearner
{
    private:
    unique_ptr<DTNode> root;
    Rand &m_rand;
    const size_t PRINTTREEMARGIN = 0;
    public:
    const int UNKNOWNCATEGORYVALUE = -1;
    const double VALIDATIONPERCENTAGE = .2;
    const char* PRINTTREEENV = "TREE";
    const char* DATALIMITENV = "DATALIMIT";

    DecisionTree(Rand &r);
    ~DecisionTree();

    size_t getNodeCount();
    size_t getTreeDepth();
    //wrapper function to train model
    void createTree(vector<tuple<vector<int>, int>> dataAndLabel);
    //wrapper function to classify new data
    int classifyData(vector<int> data);
    void printAttributeSplits();
    void pruneTree(size_t depthLimit);

    // Train the model to predict the labels
    void train(Matrix &features, Matrix &labels);
    
    // Evaluate the features and predict the labels
    void predict(const vector<double> &features, vector<double> &labels);
};

#endif