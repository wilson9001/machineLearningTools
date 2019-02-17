#include "NonInputNode.h"

void NonInputNode::createWeights()
{
    std::random_device generatorSeed;
    std::mt19937 runGenerator(generatorSeed()); //Standard mersenne_twister_engine
    std::uniform_real_distribution<> distributeInRange(WEIGHTINITLOWERBOUND, WEIGHTINITUPPERBOUND);

    weights = new vector<double>();
    (*weights).reserve(inputs.size());

    for (size_t i = 0; i < inputs.size(); i++)
    {
        (*weights).push_back(distributeInRange(runGenerator));
    }
}

NonInputNode::NonInputNode(vector<Node*> inputs): Node()
{
    this->inputs = inputs;
    learningRate = DEFAULTLEARNINGRATE;
    error = DEFAULTERROR;
    
    createWeights();
}

NonInputNode::NonInputNode(vector<Node*> inputs, double learningRate): NonInputNode(inputs)
{
    this->learningRate = learningRate;
}

NonInputNode::NonInputNode(vector<Node*> inputs, double learningRate, double error): NonInputNode(inputs, learningRate)
{
    this->error = error;
}

NonInputNode::~NonInputNode()
{
    delete weights;
}

void NonInputNode::adjustWeights(double target)
{
    //compute error
    error = (target - output)*output*(1-output);

    //adjust weights

}