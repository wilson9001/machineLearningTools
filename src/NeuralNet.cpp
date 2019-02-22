#include "NeuralNet.h"
#ifdef _DEBUG
#include <iostream>
#endif

NeuralNet::NeuralNet(Rand &r): SupervisedLearner(), m_rand(r)//, layerCount(DEFAULTLAYERCOUNT), layers(vector<shared_ptr<Layer>>())
{
    /*#ifdef _DEBUG
    cout << "Creating neural net..." << endl;
    #endif*/

    m_rand = r;
    //Add +1 for layer that interacts with inputs before first middle layer.
    middleLayerCount = DEFAULTMIDDLELAYERCOUNT + 1;
    layers = vector<shared_ptr<Layer>>();
}

/*NeuralNet::NeuralNet(size_t layerCount): layerCount(layerCount), layers(vector<shared_ptr<Layer>>())
{}*/

NeuralNet::~NeuralNet()
{}

void NeuralNet::train(Matrix &features, Matrix &labels)
{
    //first epoch network must be created
    if(layers.empty())
    {
        createNeuralNetwork(features.row(0), labels.cols());
    }

    /*#ifdef _DEBUG
    cout << "Beginning training..." << endl;
    #endif*/

    //run training
    for(size_t i = 0; i < features.rows(); i++)
    {
        layers.at(0)->setOutputs(features.row(i));

        /*#ifdef _DEBUG
        cout << "Calculating output" << endl;
        #endif*/

        //pull inputs through network
        for(shared_ptr<Layer>& layer : layers)
        {
            layer->calculateOutputs();
        }

        /*#ifdef _DEBUG
        cout << "Backpropogating error for layer:" << endl;
        #endif*/

        //propogate error back through network
        for(size_t j = (layers.size() - 1); j > 0; j--)
        {
            /*#ifdef _DEBUG
            cout << j << endl;
            #endif*/

            layers.at(j)->backPropogateError(labels.row(i));
        }
    }

    
    /*#ifdef _DEBUG
    cout << "End Training" << endl;
    #endif*/
}

void NeuralNet::predict(const vector<double> &features, vector<double> &labels)
{
    /*#ifdef _DEBUG
    cout << "Begin prediction" << endl;
    #endif*/

    vector<double> featuresCopy(features);

    layers.at(0)->setOutputs(featuresCopy);

    //pull inputs through network
    for(shared_ptr<Layer>& layer : layers)
    {
        layer->calculateOutputs();
    }

    vector<double> results = layers.back()->getOutputs();
       
    for(size_t i = 0; i < results.size(); i++)
    {
        labels.at(i) = results.at(i);
    }

    /*#ifdef _DEBUG
    cout << "End prediction" << endl;
    #endif*/
}

void NeuralNet::createNeuralNetwork(vector<double> initialInputs, size_t targetCount)
{
    /*#ifdef _DEBUG
    cout << "Creating layers..." << endl;
    #endif*/

    size_t hiddenNodeLayerSize = 2 * initialInputs.size();
    layers.clear();

    /*#ifdef _DEBUG
    cout << "Creating input layer with " << initialInputs.size() << " nodes" << endl;
    #endif*/
    
    layers.push_back(make_shared<Layer>(layerTypes::input, initialInputs.size(), nullptr, initialInputs, DEFAULTLEARNINGRATE));

    /*#ifdef _DEBUG
    cout << "Creating middle layers (with " << hiddenNodeLayerSize << " nodes in each).\nCreating middle layer:" << endl;
    #endif*/

    size_t i;
    for (i = 1; i <= middleLayerCount; i++)
    {
        /*#ifdef _DEBUG
        cout << i << endl;
        #endif*/
        
        layers.push_back(make_shared<Layer>(layerTypes::middle, hiddenNodeLayerSize, layers.back(), vector<double>(), DEFAULTLEARNINGRATE));
    }

    /*#ifdef _DEBUG
    cout << "Creating output layer with " << targetCount << " nodes" << endl;
    #endif*/

    layers.push_back(make_shared<Layer>(layerTypes::nonInput, targetCount, layers.back(), vector<double>(), DEFAULTLEARNINGRATE));
    
    /*#ifdef _DEBUG
    cout << "Creating backpointers..." << endl;
    #endif*/
    //connect backpointers
    for(; i > 0; i--)
    {
        layers.at(i-1)->setNodeOutputs(layers.at(i)->getNodes());
    }

    /*#ifdef _DEBUG
    cout << "Finished creating layers..." << endl;
    #endif*/
}