#include "MiddleNode.h"
#ifdef _DEBUG
#include <iostream>
#endif

MiddleNode::MiddleNode(vector<shared_ptr<Node>> inputs): NonInputNode(inputs), outputs(vector<shared_ptr<Node>>())
{}

MiddleNode::MiddleNode(vector<shared_ptr<Node>> inputs, double learningRate, double momentum) : NonInputNode(inputs, learningRate, momentum), outputs(vector<shared_ptr<Node>>())
{}

MiddleNode::MiddleNode(vector<shared_ptr<Node>> inputs, double learningRate, double momentum, double error) : NonInputNode(inputs, learningRate, momentum, error), outputs(vector<shared_ptr<Node>>())
{}

MiddleNode::MiddleNode(vector<shared_ptr<Node>> inputs, double learningRate, double momentum, double error, vector<shared_ptr<Node>> outputs) : NonInputNode(inputs, learningRate, momentum, error), outputs(outputs)
{}

MiddleNode::~MiddleNode()
{}

vector<shared_ptr<Node>> MiddleNode::getOutputs()
{
    return outputs;
}

void MiddleNode::setOutputs(vector<shared_ptr<Node>> outputs)
{
    #ifdef _DEBUG
    cout << "Inside setOutputs there are  " << outputs.size() << " outputs to be set" << endl;
    #endif

    this->outputs = outputs;
}

void MiddleNode::addOutput(shared_ptr<Node> output)
{
    outputs.push_back(output);
}

void MiddleNode::calculateError(double target)
{
    //TODO: Check for NULL outputs before pulling values and throw error if true.
    //TODO: Calculate error as sum(dot(error of one output node, every weight in current node))?
    if(outputs.empty())
    {
        ThrowError("Attempted to calculate error in middle node with no output connections");
    }

    double errorSum = 0;

    for(size_t i = 0; i < outputs.size(); i++)
    {
        errorSum += (outputs.at(i)->getError() * outputs.at(i)->getOldWeightForInput(uuid));
    }

    errorSum *= (output * (1-output));

    error = errorSum;
}