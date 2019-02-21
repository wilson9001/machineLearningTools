#ifndef NEURALNET_H
#define NEURALNET_H

#include "Layer.h"

class NeuralNet
{
    private:
    vector<unique_ptr<Layer>> layers;

    public:
    NeuralNet();
    NeuralNet(size_t layerCount);
    NeuralNet(size_t layerCount, size_t nodeCount);
    NeuralNet(size_t layerCount, vector<size_t> numberNodesInLayers);
    ~NeuralNet();

    void train(vector<double> trainingData, vector<double> targets);
    vector<double> predict(vector<double> inputs);
};

#endif