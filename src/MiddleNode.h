#ifndef MIDDLENODE_H
#define MIDDLENODE_H

#include "NonInputNode.h"

class MiddleNode : public NonInputNode
{
    protected:
    //unique_ptr<vector<shared_ptr<NonInputNode>>> outputs;
    vector<shared_ptr<NonInputNode>> outputs;

    virtual void calculateError();

    public:
    //MiddleNode(vector<shared_ptr<Node>>* inputs);
    //MiddleNode(vector<shared_ptr<Node>>* inputs, double learningRate);
    //MiddleNode(vector<shared_ptr<Node>>* inputs, double learningRate, double error);
    //MiddleNode(vector<shared_ptr<Node>>* inputs, double learningRate, double error, vector<shared_ptr<NonInputNode>>* outputs);
    
    MiddleNode(vector<shared_ptr<Node>> inputs);
    MiddleNode(vector<shared_ptr<Node>> inputs, double learningRate);
    MiddleNode(vector<shared_ptr<Node>> inputs, double learningRate, double error);
    MiddleNode(vector<shared_ptr<Node>> inputs, double learningRate, double error, vector<shared_ptr<NonInputNode>> outputs);
    
    virtual ~MiddleNode();

    vector<shared_ptr<NonInputNode>> getOutputs();

    //void setOutputs(vector<shared_ptr<NonInputNode>>* outputs);
    void setOutputs(vector<shared_ptr<NonInputNode>> outputs);
    void addOutput(shared_ptr<NonInputNode> output);
};

#endif