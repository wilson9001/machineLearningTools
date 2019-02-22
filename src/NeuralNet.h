#ifndef NEURALNET_H
#define NEURALNET_H

#include "Layer.h"
#include "learner.h"

class NeuralNet: public SupervisedLearner
{
    private:
    vector<shared_ptr<Layer>> layers;
    const size_t DEFAULTMIDDLELAYERCOUNT = 1;
    size_t middleLayerCount;
    const double DEFAULTLEARNINGRATE = .1;
    const size_t EPOCHWITHNOCHANGELIMIT = 5;
    const double EPOCHCHANGETHRESHOLD = .05;

    protected:
    void createNeuralNetwork(vector<double> initialInputs, size_t targetCount);
    Rand &m_rand;
    
    public:
    NeuralNet(Rand &r);
    //NeuralNet(size_t layerCount);
    //NeuralNet(size_t layerCount, size_t nodeCount);
    //NeuralNet(size_t layerCount, vector<size_t> numberNodesInLayers);
    ~NeuralNet();

    // Train the model to predict the labels
    void train(Matrix &features, Matrix &labels);
    
    // Evaluate the features and predict the labels
    void predict(const vector<double> &features, vector<double> &labels);
};

#endif