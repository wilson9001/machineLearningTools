#include "NonInputNode.h"

void NonInputNode::createWeights()
{
    random_device generatorSeed; //System-specific random number generator. Likely /dev/urandom on *nix systems.
    mt19937 runGenerator(generatorSeed()); //Standard mersenne_twister_engine
    uniform_real_distribution<> distributeInRange(WEIGHTINITLOWERBOUND, WEIGHTINITUPPERBOUND);

    //weights = make_unique<vector<double>>(); //unique_ptr(new vector<double>());
    weights = vector<double>();
    //(*weights).reserve((*inputs).size());
    weights.reserve(inputs.size());

    //for (size_t i = 0; i < (*inputs).size(); i++)
    for (size_t i = 0; i < inputs.size(); i++)
    {
        //(*weights).push_back(distributeInRange(runGenerator));
        weights.push_back(distributeInRange(runGenerator));
    }
}

//NonInputNode::NonInputNode(unique_ptr<vector<shared_ptr<Node>>>& inputs): Node()
NonInputNode::NonInputNode(vector<shared_ptr<Node>> inputs): Node()
{
    //this->inputs = move(inputs)
    this->inputs = inputs
    learningRate = DEFAULTLEARNINGRATE;
    error = DEFAULTERROR;
    
    createWeights();
}

//NonInputNode::NonInputNode(unique_ptr<vector<shared_ptr<Node>>>& inputs, double learningRate): NonInputNode(inputs)
NonInputNode::NonInputNode(vector<shared_ptr<Node>> inputs, double learningRate): NonInputNode(inputs)
{
    this->learningRate = learningRate;
}

//NonInputNode::NonInputNode(unique_ptr<vector<shared_ptr<Node>>>& inputs, double learningRate, double error): NonInputNode(inputs, learningRate)
NonInputNode::NonInputNode(vector<shared_ptr<Node>> inputs, double learningRate, double error): NonInputNode(inputs, learningRate)
{
    this->error = error;
}

NonInputNode::~NonInputNode()
{}

void NonInputNode::adjustWeights()
{
    for(size_t i = 0; i < (*weights).size(); i++)
    {
        (*weights).at(i) += ((*weights).at(i)*error*(*((*inputs).at(i))).getOutput());
    }
}

size_t NonInputNode::getInputSize()
{
    //return (*inputs).size();
    return inputs.size();
}

//virtual
void NonInputNode::calculateError(double target)
{
    error = (target - output)*output*(1-output);
}