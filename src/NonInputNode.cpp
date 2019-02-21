#include "NonInputNode.h"

void NonInputNode::createWeights()
{
    random_device generatorSeed; //System-specific random number generator. Likely /dev/urandom on *nix systems.
    mt19937 runGenerator(generatorSeed()); //Standard mersenne_twister_engine
    uniform_real_distribution<> distributeInRange(WEIGHTINITLOWERBOUND, WEIGHTINITUPPERBOUND);

    //weights = make_unique<vector<double>>(); //unique_ptr(new vector<double>());
    weights = vector<double>();
    input_weightIndex_map = unordered_map<string, size_t>();
    //(*weights).reserve((*inputs).size());
    weights.reserve(inputs.size());

    //for (size_t i = 0; i < (*inputs).size(); i++)
    /*for (size_t i = 0; i < inputs.size(); i++)
    {
        //(*weights).push_back(distributeInRange(runGenerator));
        weights.push_back(distributeInRange(runGenerator));
    }*/

    //for(const shared_ptr<Node>& inputPtr : inputs)
    for (size_t i = 0; i < inputs.size(); i++)
    {
        double newWeight = distributeInRange(runGenerator);
        weights.push_back(newWeight);
        oldWeights.push_back(newWeight);

        //input_weightIndex_map.emplace((*(inputs.at(i))).getUUID(), i);
        input_weightIndex_map.emplace(inputs.at(i)->getUUID(), i);
    }
}

//NonInputNode::NonInputNode(unique_ptr<vector<shared_ptr<Node>>>& inputs): Node()
NonInputNode::NonInputNode(vector<shared_ptr<Node>> inputs): Node(), inputs(inputs), learningRate(DEFAULTLEARNINGRATE), error(DEFAULTERROR), momentum(DEFAULTMOMENTUM)
{
    //this->inputs = move(inputs)
    //this->inputs = inputs;
    //learningRate = DEFAULTLEARNINGRATE;
    //error = DEFAULTERROR;
    //this->momentum = momentum;

    createWeights();
}

//NonInputNode::NonInputNode(unique_ptr<vector<shared_ptr<Node>>>& inputs, double learningRate): NonInputNode(inputs)
NonInputNode::NonInputNode(vector<shared_ptr<Node>> inputs, double learningRate, double momentum): NonInputNode(inputs)//, learningRate(learningRate), momentum(momentum)
{
    this->learningRate = learningRate;
    this->momentum = momentum;
}

//NonInputNode::NonInputNode(unique_ptr<vector<shared_ptr<Node>>>& inputs, double learningRate, double error): NonInputNode(inputs, learningRate)
NonInputNode::NonInputNode(vector<shared_ptr<Node>> inputs, double learningRate, double momentum, double error): NonInputNode(inputs, learningRate, momentum)//, error(error)
{
    this->error = error;
}

NonInputNode::~NonInputNode()
{}

void NonInputNode::adjustWeights()
{
    //for(size_t i = 0; i < (*weights).size(); i++)

    double previousWeightChange = 0;
    double weightChange;
    
    for(size_t i = 0; i < weights.size(); i++)
    {
        //(*weights).at(i) += ((*weights).at(i)*error*(*((*inputs).at(i))).getOutput());
        //weights.at(i) += (weights.at(i)*error*((*(inputs.at(i))).getOutput()));
        oldWeights.at(i) = weights.at(i);

        weightChange = (momentum and i > 0) ? 
        (learningRate * error * inputs.at(i)->getOutput()) + (momentum * previousWeightChange) 
        : (learningRate * error * (inputs.at(i)->getOutput()));

       previousWeightChange = weightChange;

        weights.at(i) += weightChange;
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
    error = (target - output) * output * (1 - output);
}

double NonInputNode::getError()
{
    return error;
}

double NonInputNode::getOldWeightForInput(string inputNodeUUID)
{
    //find weight from map and return weight.
    size_t weightIndex = input_weightIndex_map.at(inputNodeUUID);

    return oldWeights.at(weightIndex);
}

void NonInputNode::calculateOutput()
{
    double net = 0;

    if(inputs.empty())
    {
        ThrowError("Attempted to calculate node output with no inputs");
    }

    for(shared_ptr<Node>& input : inputs)
    {
        net += input->getOutput();
    }

    net *= -1;

    output = 1/(1+exp(net));
}

double NonInputNode::getMomentum()
{
    return momentum;
}