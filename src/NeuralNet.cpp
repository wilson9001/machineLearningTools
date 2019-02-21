#include "NeuralNet.h"

NeuralNet::NeuralNet(Rand &r): SupervisedLearner(), m_rand(r)//, layerCount(DEFAULTLAYERCOUNT), layers(vector<shared_ptr<Layer>>())
{
    m_rand = r;
    layerCount = DEFAULTLAYERCOUNT;
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
        createNeuralNetwork(features.row(0), labels.row(0).size());
    }

    //run training
    for(size_t i = 0; i < features.rows(); i++)
    {
        layers.at(0)->setOutputs(features.row(i));

        //pull inputs through network
        for(shared_ptr<Layer>& layer : layers)
        {
            layer->calculateOutputs();
        }

        //propogate error back through network
        for(size_t j = (layers.size() - 1); j >= 0; j--)
        {
            layers.at(j)->backPropogateError(labels.row(i));
        }
    }
}

void NeuralNet::predict(const vector<double> &features, vector<double> &labels)
{
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
}

void NeuralNet::createNeuralNetwork(vector<double> initialInputs, size_t targetCount)
{
    size_t hiddenNodeLayerSize = 2 * initialInputs.size();
    layers.clear();

    size_t middleLayers = layerCount - 2;

    layers.push_back(make_shared<Layer>(layerTypes::input, initialInputs.size(), nullptr, initialInputs, DEFAULTLEARNINGRATE));

    size_t i;
    for (i = 1; i <= middleLayers; i++)
    {
        layers.push_back(make_shared<Layer>(layerTypes::middle, hiddenNodeLayerSize, layers.back(), vector<double>(), DEFAULTLEARNINGRATE));
    }

    layers.push_back(make_shared<Layer>(layerTypes::nonInput, targetCount, layers.back(), vector<double>(), DEFAULTLEARNINGRATE));
    
    //connect backpointers
    for(++i; i > 0; i--)
    {
        layers.at(i-1)->setNodeOutputs(layers.at(i)->getNodes());
    }
}