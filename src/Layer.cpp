#include "Layer.h"
#ifdef _DEBUG
#include <iostream>
#endif

Layer::Layer(layerTypes layerType, size_t nodeCount, shared_ptr<Layer> previousLayer, vector<double> initialOutputs, double learningRate): layerType(layerType)
{
    nodes = make_unique<vector<shared_ptr<Node>>>();
    bias = make_shared<Node>(DEFAULTBIASVALUE);
    nodes->reserve(nodeCount);
    vector<shared_ptr<Node>> inputs;

    double momentum;

    char* momentumEnv = getenv("momentum");

    if(momentumEnv)
    {
        string momentumEnvString = momentumEnv;
        momentum = stod(momentumEnvString);
    }
    else
    {
        momentum = DEFAULTMOMENTUM;
    }

    //ofstream logFile;

    //create nodes
    switch (layerType){
        case layerTypes::middle:
            inputs = previousLayer->getNodes();
            inputs.push_back(bias);
            for(size_t i = 0; i < nodeCount; i++)
            {
                nodes->push_back(make_shared<MiddleNode>(inputs, learningRate, momentum));
            }

            break;

        case layerTypes::input:

            /*logFile.open("accuracyResults.csv", ios_base::app);

            logFile << momentum << ",";

            logFile.close();*/

            if(nodeCount != initialOutputs.size())
            {
                ThrowError("Number of initial outputs provided != number of output nodes requested");
            }

            nodes->reserve(nodeCount);

            for (double initialOutput : initialOutputs)
            {
                nodes->push_back(make_shared<Node>(initialOutput));
            }

            break;

        case layerTypes::nonInput:
        
            inputs = previousLayer->getNodes();
            inputs.push_back(bias);

            for (size_t i = 0; i < nodeCount; i++)
            {
                nodes->push_back(make_shared<NonInputNode>(inputs, learningRate, momentum));
            }
    }
}

Layer::Layer(layerTypes layerType, size_t nodeCount, shared_ptr<Layer> previousLayer, vector<double> initialOutputs, double learningRate, double bias): Layer(layerType, nodeCount, previousLayer, initialOutputs, learningRate)
{
    this->bias->setOutput(bias);
    //this->bias = make_shared<Node>(bias);
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
        #ifdef _DEBUG
        //cout << "\nCalculating error for node " << i << endl;
        #endif
        double target = i < targets.size() ? targets.at(i) : targets.back();
        nodes->at(i)->calculateError(target);
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
    if(outputs.size() != nodes->size())
    {
        ThrowError("Number of outputs to set != number of nodes in layer");
    }

    #ifdef _DEBUG
    //cout << "\n--------------------------------\nSetting outputs in layer\n";
    #endif

    for(size_t i = 0; i < outputs.size(); i++)
    {
        #ifdef _DEBUG
        //cout << "old output: " << nodes->at(i)->getOutput() << " potential new output: " << outputs.at(i);
        #endif

        nodes->at(i)->setOutput(outputs.at(i));

        #ifdef _DEBUG
        //cout << " actual new output: " << nodes->at(i)->getOutput() << endl;
        #endif
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

void Layer::setNodeOutputs(vector<shared_ptr<Node>> nodeOutputs)
{
    /*#ifdef _DEBUG
    if(nodeOutputs.empty())
    {
        cout << "WARNING: SETTING EMPTY VECTOR AS NODE OUTPUTS IN A LAYER" << endl;
    }
    else
    {
        cout << "Setting " << nodeOutputs.size() << " nodes as outputs" << endl;
    }
    #endif*/

    /*for(shared_ptr<Node>& node : *nodes)
    {
        node->setOutputs(nodeOutputs);
    }*/
    for(size_t i = 0; i < nodes->size(); i++)
    {
        nodes->at(i)->setOutputs(nodeOutputs);
    }
}

void Layer::setWeights(vector<vector<double>> newWeights)
{
    if(newWeights.size() != nodes->size())
    {
        ThrowError("Set weights: number of nodes passed in != number of nodes in layer");
    }

    for(size_t i = 0; i < nodes->size(); i++)
    {
        nodes->at(i)->setWeights(newWeights.at(i));
    }
}