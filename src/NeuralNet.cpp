#include "NeuralNet.h"
#include "matrix.h"
#include <iostream>

NeuralNet::NeuralNet(Rand &r): SupervisedLearner(), m_rand(r)//, layerCount(DEFAULTLAYERCOUNT), layers(vector<shared_ptr<Layer>>())
{
    /*#ifdef _DEBUG
    cout << "Creating neural net..." << endl;
    #endif*/

    m_rand = r;

    char* hiddenLayersEnv = getenv("hiddenLayers");

    if(hiddenLayersEnv)
    {
        string hiddenLayersEnvString = hiddenLayersEnv;
        middleLayerCount = static_cast<size_t>(stol(hiddenLayersEnvString));
    }
    else
    {
       middleLayerCount = DEFAULTMIDDLELAYERCOUNT;
    }

    ofstream logFile("accuracyResults.csv", ios_base::app);

    logFile << middleLayerCount << ",";

    logFile.close();

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
    Matrix validationSetFeatures(features);
    Matrix validationSetLabels(labels);
    Matrix validationSetLabelsContinuous(labels);

    Matrix trainingSetFeatures(features);
    Matrix trainingSetLabels(labels);
    
    Matrix trainingSetFeaturesUnmixed(features);
    Matrix trainingSetLabelsContinuous(labels);

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
        validationSetLabelsContinuous.copyRow(labels.row(i));
    }

    #ifdef _DEBUG
    cout << "creating training set with " << (features.rows() - validationSetSize) << " rows" << endl;
    #endif

    for(; i < features.rows(); i++)
    {
        trainingSetFeatures.copyRow(features.row(i));
        trainingSetFeaturesUnmixed.copyRow(features.row(i));
        trainingSetLabels.copyRow(labels.row(i));
        trainingSetLabelsContinuous.copyRow(labels.row(i));
    }

    validationSetLabelsContinuous.makeNominalAttrContinuous(0);
    trainingSetLabelsContinuous.makeNominalAttrContinuous(0);

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
    double currentEpochMeanSquaredError = 1;
    double bestEpochAccuracy = 0;
    double bestEpochMeanSquaredError = 100;
    vector<vector<vector<double>>> bestWeightConfiguration(getAllWeights());
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
            /*vector<double> results = layers.back()->getOutputs();
       
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

            cout << "Output: " << categoryForAnswer;*/
            #endif

            vector<double> nominalizedResults = vector<double>(DEFAULTOUTPUTNODECOUNT, 0);

            size_t indexToChange = static_cast<size_t>(trainingSetLabels.row(i).at(0));
            //size_t indexToChange = static_cast<size_t>(labels.row(i).at(0));
            nominalizedResults.at(indexToChange) = 1;

            #ifdef _DEBUG
            //cout << ", target: " << indexToChange << endl;
            #endif

            //propogate error back through network
            for(size_t j = (layers.size() - 1); j > 0; j--)
            {
                layers.at(j)->backPropogateError(nominalizedResults);
            }
        }

        //cout << "\nNumber of features associated with labels: " << validationSetLabels.valueCount(0) << endl;

        currentEpochAccuracy = measureAccuracy(validationSetFeatures, validationSetLabels); //uses 3, so is nominal. Change to continuous?



        currentEpochMeanSquaredError = pow(measureAccuracy(validationSetFeatures, validationSetLabelsContinuous), 2);
        //currentEpochMeanSquaredError = measureAccuracy(trainingSetFeaturesUnmixed, trainingSetLabelsContinuous);



        //currentEpochAccuracy = measureAccuracy(features, labels);

        #ifdef _DEBUG
        double currentTrainingSetMSE = measureAccuracy(trainingSetFeaturesUnmixed, trainingSetLabelsContinuous);
        cout << "Current best VS accuracy:" << bestEpochAccuracy << " Current best VS MSE: " << bestEpochMeanSquaredError << " Current best training set MSE: " << currentTrainingSetMSE << endl;
        #endif

        if(currentEpochMeanSquaredError < bestEpochMeanSquaredError)
        {
            #ifdef _DEBUG
            cout << "improvement made\n";
            #endif
            epochsSinceImprovement = 0;
            bestEpochAccuracy = currentEpochAccuracy;
            bestEpochMeanSquaredError = currentEpochMeanSquaredError;
            
            //save current best configuration
            bestWeightConfiguration = getAllWeights();
        }
        else
        {
            ++epochsSinceImprovement;
        }
        
        //changeInAccuracy = currentEpochAccuracy - bestEpochAccuracy;

        //changeInAccuracy < EPOCHCHANGETHRESHOLD ? ++epochsSinceImprovement : epochsSinceImprovement = 0;
        
        totalEpochs++;

    } while (epochsSinceImprovement < EPOCHWITHNOIMPROVEMENTLIMIT /*|| bestEpochAccuracy <= .8*/);

    setAllWeights(bestWeightConfiguration);

    double finalTrainingSetMSE = measureAccuracy(trainingSetFeaturesUnmixed, trainingSetLabelsContinuous);

    cout << "Final VS accuracy:" << bestEpochAccuracy << endl << "Final VS MSE: " << bestEpochMeanSquaredError << endl << "Final training set MSE: " << finalTrainingSetMSE << endl;

    ofstream logFile("accuracyResults.csv", ios_base::app);

    logFile /*<< DEFAULTLEARNINGRATE << "," <<  EPOCHWITHNOIMPROVEMENTLIMIT << ","*/ << finalTrainingSetMSE << ","  << bestEpochMeanSquaredError << ",";

    logFile.close();

    cout << "Total training epochs: " << totalEpochs << endl;
}

void NeuralNet::predict(const vector<double> &features, vector<double> &labels)
{
    /*#ifdef _DEBUG
    cout << "Begin prediction" << endl;
    #endif*/

    vector<double> featuresCopy(features);
    vector<double> labelsCopyContinuous(labels);

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

    size_t hiddenNodeLayerSize;
    
    char* hiddenNodesEnv = getenv("hiddenLayerSize");

    if(hiddenNodesEnv)
    {
        string hiddenNodesEnvString = hiddenNodesEnv;
        hiddenNodeLayerSize = static_cast<size_t>(stol(hiddenNodesEnvString));
    }
    else
    {
        hiddenNodeLayerSize = 2 * initialInputs.size();
    }

    /*ofstream logFile("accuracyResults.csv", ios_base::app);

    logFile << hiddenNodeLayerSize << ",";

    logFile.close();*/

    layers.clear();

    #ifdef _DEBUG
    cout << "Creating input layer with " << initialInputs.size() << " nodes" << endl;
    #endif
    
    layers.push_back(make_shared<Layer>(layerTypes::input, initialInputs.size(), nullptr, initialInputs, DEFAULTLEARNINGRATE));

    #ifdef _DEBUG
    cout << "Creating " << middleLayerCount << " middle layers (with " << hiddenNodeLayerSize << " nodes in each).\nCreating middle layer:" << endl;
    #endif

    size_t i;
    for (i = 1; i <= middleLayerCount; i++)
    {
        #ifdef _DEBUG
        //cout << i << endl;
        #endif
        
        if(i > 1)
        {
            layers.push_back(make_shared<Layer>(layerTypes::middle, hiddenNodeLayerSize, layers.back(), vector<double>(), DEFAULTLEARNINGRATE));
        }
        else
        {
            layers.push_back(make_shared<Layer>(layerTypes::middle, initialInputs.size(), layers.back(), vector<double>(), DEFAULTLEARNINGRATE));
        }
    }

    #ifdef _DEBUG
    cout << "Creating output layer with " << targetCount << " nodes" << endl;
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

vector<vector<vector<double>>> NeuralNet::getAllWeights()
{
    vector<vector<vector<double>>> weights;

    for(shared_ptr<Layer>& layer : layers)
    {
        weights.push_back(layer->getWeights());
    }

    return weights;
}

void NeuralNet::setAllWeights(vector<vector<vector<double>>> allWeights)
{
    if(allWeights.size() != layers.size())
    {
        //cerr << "Set all weights: number of layers passed in, number of layers in network: " << allWeights.size() << " " << layers.size();
        ThrowError("Set all weights: number of layers passed in, number of layers in network:", to_str(allWeights.size()), ", ", to_str(layers.size()));
    }

    for(size_t i = 0; i < layers.size(); i++)
    {
        layers.at(i)->setWeights(allWeights.at(i));
    }
}