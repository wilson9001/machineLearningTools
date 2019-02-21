#include "MiddleNode.h"

//MiddleNode::MiddleNode(vector<shared_ptr<Node>>* inputs): NonInputNode(inputs)
MiddleNode::MiddleNode(vector<shared_ptr<Node>> inputs): NonInputNode(inputs)
{
    //outputs = NULL;
    outputs = vector<shared_ptr<NonInputNode>>();
}

//MiddleNode::MiddleNode(vector<shared_ptr<Node>>* inputs, double learningRate) : NonInputNode(inputs, learningRate)
MiddleNode::MiddleNode(vector<shared_ptr<Node>> inputs, double learningRate) : NonInputNode(inputs, learningRate)
{
    //outputs = NULL;
    outputs = vector<shared_ptr<NonInputNode>>();
}

//MiddleNode::MiddleNode(vector<shared_ptr<Node>>* inputs, double learningRate, double error) : NonInputNode(inputs, learningRate, error)
MiddleNode::MiddleNode(vector<shared_ptr<Node>> inputs, double learningRate, double error) : NonInputNode(inputs, learningRate, error)
{
    //outputs = NULL;
    outputs = vector<shared_ptr<NonInputNode>>();
}

//MiddleNode::MiddleNode(vector<shared_ptr<Node>>* inputs, double learningRate, double error, vector<shared_ptr<NonInputNode>>* outputs) : NonInputNode(inputs, learningRate, error)
MiddleNode::MiddleNode(vector<shared_ptr<Node>> inputs, double learningRate, double error, vector<shared_ptr<NonInputNode>> outputs) : NonInputNode(inputs, learningRate, error)
{
    //this->outputs = unique_ptr(outputs);
    this->outputs = outputs;
}

MiddleNode::~MiddleNode()
{
    //outputs = NULL;
}

vector<shared_ptr<NonInputNode>>& MiddleNode::getOutputs()
{
    //return *outputs;
    return outputs;
}

void MiddleNode::setOutputs(vector<shared_ptr<NonInputNode>> outputs)
{
    //this->outputs.reset();
    //this->outputs = unique_ptr(outputs);

    this->outputs = outputs;
}

void MiddleNode::addOutput(shared_ptr<NonInputNode> output)
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
        //errorSum += ((*(outputs.at(i))).getError() * weights.at(i));
        errorSum += (outputs.at(i)->getError() * outputs.at(i)->getOldWeightForInput(uuid));
    }

    errorSum *= (output * (1-output));

    error = errorSum;
}