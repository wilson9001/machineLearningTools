#ifndef NONINPUTNODE_H
#define NONINPUTNODE_H

#include "Node.h"
#include <vector>
#include <random>
#include <memory>

using namespace std;

class NonInputNode: Node
{
    protected:
    //unique_ptr<vector<double>> weights;
    //unique_ptr<vector<shared_ptr<Node>>> inputs;
    vector<double> weights;
    vector<shared_ptr<Node>> inputs;
    
    double error;
    double learningRate;
    const double DEFAULTLEARNINGRATE = .1;
    const double DEFAULTERROR = 0;
    const double WEIGHTINITLOWERBOUND = -.5;
    const double WEIGHTINITUPPERBOUND = .5;

    void adjustWeights();
    void createWeights();
    virtual void calculateError(double target);
    
    public:
    // NonInputNode(unique_ptr<vector<shared_ptr<Node>>>& inputs);
    // NonInputNode(unique_ptr<vector<shared_ptr<Node>>>& inputs, double learningRate);
    // NonInputNode(unique_ptr<vector<shared_ptr<Node>>>& inputs, double learningRate, double error);

    NonInputNode(vector<shared_ptr<Node>> inputs);
    NonInputNode(vector<shared_ptr<Node>> inputs, double learningRate);
    NonInputNode(vector<shared_ptr<Node>> inputs, double learningRate, double error);
    
    virtual ~NonInputNode();

    vector<double> getWeights();
    vector<shared_ptr<Node>> getInputs();
    size_t getInputSize();
};

#endif