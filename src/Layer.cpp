#include "Layer.h"

Layer::Layer(layerTypes layerType, size_t nodeCount, shared_ptr<Layer> previousLayer, vector<double> initialOutputs, double learningRate): layerType(layerType)
{
    nodes = make_unique<vector<shared_ptr<Node>>>();
    nodes->reserve(nodeCount);

    //create nodes
    switch (layerType){
        case layerTypes::middle:
            
            for(size_t i = 0; i < nodeCount; i++)
            {
                nodes->push_back(make_shared<MiddleNode>(previousLayer->getNodes(), learningRate, DEFAULTMOMENTUM));
            }

            break;

        case layerTypes::input:

            if(nodeCount != initialOutputs.size())
                {
                    ThrowError("Number of initial outputs provided != number of output nodes requested");
                }

            for (double initialOutput : initialOutputs)
            {
                nodes->push_back(make_shared<Node>(initialOutput));
            }

            break;

        case layerTypes::nonInput:
            const vector<shared_ptr<Node>>& inputs = previousLayer->getNodes();

            for (size_t i = 0; i < nodeCount; i++)
            {
                nodes->push_back(make_shared<NonInputNode>(previousLayer->getNodes(), learningRate, DEFAULTMOMENTUM));
            }
    }

    bias = make_unique<Node>(DEFAULTBIASVALUE);
}

Layer::Layer(layerTypes layerType, size_t nodeCount, shared_ptr<Layer> previousLayer, vector<double> initialOutputs, double learningRate, double bias): Layer(layerType, nodeCount, previousLayer, initialOutputs, learningRate)
{
    this->bias.reset();
    this->bias = make_unique<Node>(bias);
}

Layer::~Layer()
{}

void Layer::calculateOutputs()
{
    for(shared_ptr<Node>& node : *nodes)
    {
        node->calculateOutput();
    }
}

void Layer::backPropogateError(vector<double>& targets)
{
    for (size_t i = 0; i < nodes->size(); i++)
    {
        nodes->at(i)->calculateError(targets.at(i));
        nodes->at(i)->adjustWeights();
    }
}

Node& Layer::getBias()
{
    return *bias;
}

layerTypes Layer::getLayerType()
{
    return layerType;
}

vector<shared_ptr<Node>> Layer::getNodes()
{
    return *nodes;
}

vector<vector<double>> Layer::getWeights()
{
    vector<vector<double>> weights = vector<vector<double>>();

    weights.reserve(nodes->size());

    for(shared_ptr<Node>& node : *nodes)
    {
        weights.push_back(node->getWeights());
    }

    return weights;
}

void Layer::setOutputs(vector<double>& outputs)
{
    if(outputs.size != nodes->size())
    {
        ThrowError("Number of outputs to set != number of nodes in layer");
    }

    for(size_t i = 0; i < outputs.size(); i++)
    {
        nodes->at(i)->setOutput(outputs.at(i));
    }
}

vector<double> Layer::getOutputs()
{
    vector<double> outputs = vector<double>();
    outputs.reserve(nodes->size());

    for (shared_ptr<Node>& node : *nodes)
    {
        outputs.push_back(node->getOutput());
    }

    return outputs;
}

void Layer::setBias(double newBias)
{
    bias->setOutput(newBias);
}

size_t Layer::getNodeCount()
{
    return nodes->size();
}