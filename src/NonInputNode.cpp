#include "NonInputNode.h"

void NonInputNode::createWeights()
{
    random_device generatorSeed; //System-specific random number generator. Likely /dev/urandom on *nix systems.
    mt19937 runGenerator(generatorSeed()); //Standard mersenne_twister_engine
    uniform_real_distribution<> distributeInRange(WEIGHTINITLOWERBOUND, WEIGHTINITUPPERBOUND);

    weights = vector<double>();
    input_weightIndex_map = unordered_map<string, size_t>();
    weights.reserve(inputs.size());

    for (size_t i = 0; i < inputs.size(); i++)
    {
        double newWeight = distributeInRange(runGenerator);
        weights.push_back(newWeight);
        oldWeights.push_back(newWeight);

        input_weightIndex_map.emplace(inputs.at(i)->getUUID(), i);
    }
}

NonInputNode::NonInputNode(vector<shared_ptr<Node>> inputs): Node()
{
    this->inputs = inputs;
    learningRate = DEFAULTLEARNINGRATE;
    error = DEFAULTERROR;
    momentum = DEFAULTMOMENTUM;

    createWeights();
}

NonInputNode::NonInputNode(vector<shared_ptr<Node>> inputs, double learningRate, double momentum): NonInputNode(inputs)
{
    this->learningRate = learningRate;
    this->momentum = momentum;
}

NonInputNode::NonInputNode(vector<shared_ptr<Node>> inputs, double learningRate, double momentum, double error): NonInputNode(inputs, learningRate, momentum)//, error(error)
{
    this->error = error;
}

NonInputNode::~NonInputNode()
{}

void NonInputNode::adjustWeights()
{
    double previousWeightChange = 0;
    double weightChange;
    
    #ifdef _DEBUG
    //cout << "\nWeight changes for output node are: ";
    #endif

    for(size_t i = 0; i < weights.size(); i++)
    {
        #ifdef _DEBUG
        //cout << " old: " << oldWeights.at(i);
        #endif

        oldWeights.at(i) = weights.at(i);

        weightChange = (learningRate * error * inputs.at(i)->getOutput()) + (momentum * previousWeightChange);

        previousWeightChange = weightChange;

        #ifdef _DEBUG
        //cout << " weight change: " << weightChange;
        #endif

        weights.at(i) += weightChange;

        #ifdef _DEBUG
        if(oldWeights.at(i) == weights.at(i))
        {
        //    cout << ", but weight was not changed.";
        }
        else
        {
        //    cout << ", and weight was changed.";
        }
        
        //cout << " new: " << weights.at(i);
        #endif
    }

    #ifdef _DEBUG
    //cout << endl << endl;
    #endif
}

size_t NonInputNode::getInputSize()
{
    return inputs.size();
}

//virtual
void NonInputNode::calculateError(double target)
{
    error = (target - output) * output * (1 - output);

    #ifdef _DEBUG
    //cout << "Target, output, error for output node is " << target << ", " << output << ", " << error << endl;
    #endif
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
        ThrowError("Calculate output: attempted to calculate node output with no inputs");
    }

    #ifdef _DEBUG
    //cout << "Calculating output of node. Inputs, weights are: ";
    #endif

    /*for(shared_ptr<Node>& input : inputs)
    {
        net += input->getOutput();
        #ifdef _DEBUG
        cout << input->getOutput() << " ";//weights aren't used?
        #endif
    }*/

    for(size_t i = 0; i < inputs.size(); i++)
    {
        net += (inputs.at(i)->getOutput() * weights.at(i));
        #ifdef _DEBUG
        //cout << "{" << inputs.at(i)->getOutput() << ", " << weights.at(i) << "} ";
        #endif
    }

    #ifdef _DEBUG
    //cout << " final net: " << net << endl;
    #endif

    net *= -1;

    output = 1/(1+exp(net));

    #ifdef _DEBUG
    //cout << "output: " << output << endl;
    #endif
}

double NonInputNode::getMomentum()
{
    return momentum;
}

vector<double> NonInputNode::getWeights()
{
    return weights;
}

vector<shared_ptr<Node>> NonInputNode::getInputs()
{
    return inputs;
}

void NonInputNode::useMomentum(double momentum)
{
    this->momentum = momentum;
}

void NonInputNode::setWeights(vector<double> newWeights)
{
    if(newWeights.size() != weights.size())
    {
        ThrowError("Set weights: new weight count and weight count don't match");
    }

    for(size_t i = 0; i < weights.size(); i++)
    {
        weights.at(i) = newWeights.at(i);
    }
}