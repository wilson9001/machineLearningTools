#ifndef NONINPUTNODE_H
#define NONINPUTNODE_H

#include "Node.h"
#include <vector>
#include <random>

using std::vector;

class NonInputNode: Node
{
    private:
    vector<double>* weights;
    vector<Node*> inputs;
    double error;
    double learningRate;
    const double DEFAULTLEARNINGRATE = .1;
    const double DEFAULTERROR = 0;
    const double WEIGHTINITLOWERBOUND = -.5;
    const double WEIGHTINITUPPERBOUND = .5;

    protected:
    virtual void adjustWeights(double target);
    void createWeights();

    public:
    NonInputNode(vector<Node*> inputs);
    NonInputNode(vector<Node*> inputs, double learningRate);
    NonInputNode(vector<Node*> inputs, double learningRate, double error);
    ~NonInputNode();

    vector<double> getWeights();
    vector<Node*> getInputs();
    size_t getInputSize();
};

#endif