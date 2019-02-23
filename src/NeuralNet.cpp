#include "NeuralNet.h"
#include "matrix.h"
#ifdef _DEBUG
#include <iostream>
#endif

NeuralNet::NeuralNet(Rand &r): SupervisedLearner(), m_rand(r)//, layerCount(DEFAULTLAYERCOUNT), layers(vector<shared_ptr<Layer>>())
{
    /*#ifdef _DEBUG
    cout << "Creating neural net..." << endl;
    #endif*/

    m_rand = r;

    middleLayerCount = DEFAULTMIDDLELAYERCOUNT;

    //Add +1 for layer that interacts with inputs before first middle layer.
    middleLayerCount++;

    #ifdef _DEBUG
    //cout << "Created neural net with middle layers set to " << middleLayerCount << endl;
    #endif

    layers = vector<shared_ptr<Layer>>();
}

/*NeuralNet::NeuralNet(size_t layerCount): layerCount(layerCount), layers(vector<shared_ptr<Layer>>())
{}*/

NeuralNet::~NeuralNet()
{}

void NeuralNet::train(Matrix &features, Matrix &labels)
{
    Matrix validationSetFeatures(features);// = Matrix();
    Matrix validationSetLabels(labels);// = Matrix();

    Matrix trainingSetFeatures(features);// = Matrix();
    Matrix trainingSetLabels(labels); // = Matrix();

    //create validation set
    size_t validationSetSize = static_cast<size_t>(features.rows() * VALIDATIONSETPERCENTAGE);

    size_t i = 0;

    features.shuffleRows(m_rand, &labels);

    #ifdef _DEBUG
    cout << "creating validation set with " << validationSetSize << " rows" << endl;
    #endif

    for (; i < validationSetSize; i++)
    {
        #ifdef _DEBUG
        //cout << "creating validation feature" << endl;
        #endif
        validationSetFeatures.copyRow(features.row(i));

        #ifdef _DEBUG
        //cout << "creating validation label" << endl;
        #endif
        validationSetLabels.copyRow(labels.row(i));
    }

    #ifdef _DEBUG
    cout << "creating training set with " << (features.rows() - validationSetSize) << " rows" << endl;
    #endif

    for(; i < features.rows(); i++)
    {
        trainingSetFeatures.copyRow(features.row(i));
        trainingSetLabels.copyRow(labels.row(i));
    }

    //first epoch network must be created
    if(layers.empty())
    {
        #ifdef _DEBUG
        //cout << "creating neural net" << endl;
        #endif

        createNeuralNetwork(trainingSetFeatures.row(0), DEFAULTOUTPUTNODECOUNT);
        //createNeuralNetwork(features.row(0), DEFAULTOUTPUTNODECOUNT);
    }

    #ifdef _DEBUG
    //cout << "There are " << trainingSetFeatures.rows() << " rows of training data" << endl;
    #endif
    size_t totalEpochs = 0;
    size_t epochsSinceImprovement = 0;
    double currentEpochAccuracy = 0;
    double previousEpochAccuracy = 0;
    //double changeInAccuracy = 0;

    //trainingSetFeatures.shuffleRows(m_rand, &trainingSetLabels);

    //run training epochs
    do
    {
        trainingSetFeatures.shuffleRows(m_rand, &trainingSetLabels);
        //features.shuffleRows(m_rand, &labels);

        //run 1 epoch of training
        for(size_t i = 0; i < trainingSetFeatures.rows(); i++)
        //for(size_t i = 0; i < features.rows(); i++)
        {
            layers.at(0)->setOutputs(trainingSetFeatures.row(i));
            //layers.at(0)->setOutputs(features.row(i));

            #ifdef _DEBUG
            /*cout << "training on row " << i << ": ";
            for(double input : features.row(i))
            {
                cout << input << " ";
            }
            cout << endl;*/
            #endif

            //pull inputs through network
            for(shared_ptr<Layer>& layer : layers)
            {
                layer->calculateOutputs();
            }

            #ifdef _DEBUG
            vector<double> results = layers.back()->getOutputs();
       
            size_t categoryForAnswer = 0;
            double greatestCertainty = 0;

            for(size_t i = 0; i < results.size(); i++)
            {
                if(results.at(i) > greatestCertainty)
                {
                    categoryForAnswer = i;
                    greatestCertainty = results.at(i);
                }
            }

            cout << "Output: " << categoryForAnswer;
            #endif

            vector<double> nominalizedResults = vector<double>(DEFAULTOUTPUTNODECOUNT, 0);
                
            size_t indexToChange = static_cast<size_t>(trainingSetLabels.row(i).at(0));
            //size_t indexToChange = static_cast<size_t>(labels.row(i).at(0));
            nominalizedResults.at(indexToChange) = 1;

            #ifdef _DEBUG
            cout << ", target: " << indexToChange << endl;
            #endif

            //propogate error back through network
            for(size_t j = (layers.size() - 1); j > 0; j--)
            {
                layers.at(j)->backPropogateError(nominalizedResults);
            }
        }

        currentEpochAccuracy = measureAccuracy(validationSetFeatures, validationSetLabels);
        //currentEpochAccuracy = measureAccuracy(features, labels);

        #ifdef _DEBUG
        cout << "Accuracy of epoch " << totalEpochs << ": " << currentEpochAccuracy << endl;
        #endif

        currentEpochAccuracy > previousEpochAccuracy ? epochsSinceImprovement = 0 : ++epochsSinceImprovement;
        //changeInAccuracy = currentEpochAccuracy - previousEpochAccuracy;

        //changeInAccuracy < EPOCHCHANGETHRESHOLD ? ++epochsSinceImprovement : epochsSinceImprovement = 0;
        
        totalEpochs++;
        previousEpochAccuracy = currentEpochAccuracy;

    } while (epochsSinceImprovement < EPOCHWITHNOIMPROVEMENTLIMIT /*|| currentEpochAccuracy <= .8*/);

    #ifdef _DEBUG
    cout << "Total training epochs: " << totalEpochs << endl;
    #endif
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
       
    size_t categoryForAnswer = 0;
    double greatestCertainty = 0;

    for(size_t i = 0; i < results.size(); i++)
    {
        if(results.at(i) > greatestCertainty)
        {
            categoryForAnswer = i;
            greatestCertainty = results.at(i);
        }
    }

    /*for(size_t i = 0; i < results.size(); i++)
    {
        labels.at(i) = results.at(i);
    }*/

    labels.at(0) = categoryForAnswer;

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

    #ifdef _DEBUG
    //cout << "Creating input layer with " << initialInputs.size() << " nodes" << endl;
    #endif
    
    layers.push_back(make_shared<Layer>(layerTypes::input, initialInputs.size(), nullptr, initialInputs, DEFAULTLEARNINGRATE));

    #ifdef _DEBUG
    //cout << "Creating " << middleLayerCount << " middle layers (with " << hiddenNodeLayerSize << " nodes in each).\nCreating middle layer:" << endl;
    #endif

    size_t i;
    for (i = 1; i <= middleLayerCount; i++)
    {
        #ifdef _DEBUG
        //cout << i << endl;
        #endif
        
        layers.push_back(make_shared<Layer>(layerTypes::middle, hiddenNodeLayerSize, layers.back(), vector<double>(), DEFAULTLEARNINGRATE));
    }

    #ifdef _DEBUG
    //cout << "Creating output layer with " << targetCount << " nodes" << endl;
    #endif

    layers.push_back(make_shared<Layer>(layerTypes::nonInput, targetCount, layers.back(), vector<double>(), DEFAULTLEARNINGRATE));
    
    #ifdef _DEBUG
    //cout << "Creating backpointers..." << endl;
    #endif
    //connect backpointers
    for(; i > 0; i--)
    {
        layers.at(i-1)->setNodeOutputs(layers.at(i)->getNodes());
    }

    /*#ifdef _DEBUG
    cout << "Finished creating layers..." << endl;
    #endif*/
}