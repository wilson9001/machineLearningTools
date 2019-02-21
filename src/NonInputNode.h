#ifndef NONINPUTNODE_H
#define NONINPUTNODE_H

#include "Node.h"
#include <random>
#include <unordered_map>
#include <math.h>

//sudo apt install uuid-dev installs this. Other option is attempt to read from /proc/sys/kernel/random/uuid, which is what I'm currently doing.
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
    double momentum;
    double error;
    double learningRate;
    double momentum;
    const double DEFAULTLEARNINGRATE = .1;
    const double DEFAULTERROR = 0;
    const double DEFAULTMOMENTUM = 0;
    const double WEIGHTINITLOWERBOUND = -.5;
    const double WEIGHTINITUPPERBOUND = .5;

    void createWeights();
    
    public:
    // NonInputNode(unique_ptr<vector<shared_ptr<Node>>>& inputs);
    // NonInputNode(unique_ptr<vector<shared_ptr<Node>>>& inputs, double learningRate);
    // NonInputNode(unique_ptr<vector<shared_ptr<Node>>>& inputs, double learningRate, double error);

    NonInputNode(vector<shared_ptr<Node>> inputs);
    NonInputNode(vector<shared_ptr<Node>> inputs, double learningRate, double momentum);
    NonInputNode(vector<shared_ptr<Node>> inputs, double learningRate, double momentum, double error);
    
    virtual ~NonInputNode();

    virtual void calculateError(double target);
    vector<double> getWeights();
    double getError();
    vector<shared_ptr<Node>> getInputs();
    size_t getInputSize();
    double getOldWeightForInput(string inputNodeUUID);
    void useMomentum(double momentum);
    void adjustWeights();
    void calculateOutput();
    double getMomentum();
};

#endif