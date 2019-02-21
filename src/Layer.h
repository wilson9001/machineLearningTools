#ifndef LAYER_H
#define LAYER_H

#include "MiddleNode.h"

enum class layerTypes{input, middle, nonInput};

class Layer
{
    private:
    unique_ptr<vector<shared_ptr<Node>>> nodes;
    /*unique_ptr<vector<shared_ptr<MiddleNode>>> middleNodes;
    unique_ptr<vector<shared_ptr<NonInputNode>>> NonInputNodes;*/
    unique_ptr<Node> bias;
    layerTypes layerType;
    const double DEFAULTBIASVALUE = 1;
    const double DEFAULTMOMENTUM = 0;

    public:
    Layer(layerTypes layerType, size_t nodeCount, shared_ptr<Layer> previousLayer, vector<double> initialOutputs, double learningRate);
    Layer(layerTypes layerType, size_t nodeCount, shared_ptr<Layer> previousLayer, vector<double> initialOutputs, double learningRate, double bias);
    
    /*
    Layer(size_t nodeCount, layerTypes nodeType, Layer& previousLayer, vector<double>& initialOutputs);
    Layer(size_t nodeCount, layerTypes nodeType, Layer& previousLayer, vector<double>& initialOutputs, double bias);
    Layer(size_t nodeCount, layerTypes nodeType, Layer& previousLayer, vector<double>& initialOutputs, double learningRate);
    Layer(size_t nodeCount, layerTypes nodeType, Layer& previousLayer, vector<double>& initialOutputs, double learningRate, double bias);
    */
    ~Layer();

    /*void pullAndProcessInputs();
    void pullAndProcessError();
    vector<shared_ptr<Node>> getNodes();*/ //will need to cast pointer if wanting to use features not found in node.
    Node& getBias();
    vector<shared_ptr<Node>> getNodes();
    void setOutputs(vector<double>& outputs);
    void calculateOutputs();
    void backPropogateError(vector<double>& targets);
    vector<double> getOutputs();
    vector<vector<double>> getWeights();
    layerTypes getLayerType();
    void setBias(double newBias);
    size_t getNodeCount();
};

#endif