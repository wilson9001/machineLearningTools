#ifndef NONINPUTNODE_H
#define NONINPUTNODE_H

#include "Node.h"
#include <vector>
#include <random>
#include <memory>
#include <unordered_map>

//sudo apt install uuid-dev installs this. Other option is attempt to read from /proc/sys/kernel/random/uuid, but this is simpler.
//#include <uuid/uuid.h>

class NonInputNode: public Node
{
    protected:
    //unique_ptr<vector<double>> weights;
    //unique_ptr<vector<shared_ptr<Node>>> inputs;
    vector<double> weights;
    vector<double> oldWeights;
    vector<shared_ptr<Node>> inputs;
    unordered_map<string, size_t> input_weightIndex_map;

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
    double getError();
    vector<shared_ptr<Node>> getInputs();
    size_t getInputSize();
    double getOldWeightForInput(string inputNodeUUID);
};

#endif