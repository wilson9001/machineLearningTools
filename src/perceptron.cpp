#include "perceptron.h"
#include "error.h"
#include <string>
#include <iostream>
//#include <fstream>

using namespace std;

/**
 * This helper function takes the net output from processing inputs and then converts it to a 0 or 1, since this is a perceptron.
*/
int Perceptron::processOutput(const vector<double> &inputs)
{
    double net = processInput(inputs);

    //determine output
    return net > THRESHOLD;
}

/**
 * This helper function is used to process inputs to the perceptron by performing dot products.
*/
double Perceptron::dotProduct(const vector<double> &inputs)
{
    if (inputs.size() == weights.size())
    {
        double total = 0;

        for (size_t i = 0; i <= inputVectorSize; i++)
        {
            total += (inputs[i] * weights[i]);
        }

        return total;
    }
    else
    {
        ThrowError("features vector size != weight vector size! ", to_string(inputs.size()), ", ", to_string(weights.size()));
        return 0; //This is never reached but g++ issues a warning without it.
    }
}

/**
 * This function is intended to perform any low-level pre- or post-processing on each set of inputs.
*/
double Perceptron::processInput(const vector<double> &inputs)
{
    //perform input processing
    
    double net = dotProduct(inputs);

    return net;
}

/**
 * Adjust weights by subtractig the target from the output, multiply by the learning rate and the existing weight value.
*/
void Perceptron::adjustWeights(double target, vector<double> &inputs, double output)
{
    for (size_t i = 0; i <= inputVectorSize; i++)
    {
        weights.at(i) += ((target - output) * LEARNING_RATE * inputs.at(i));
    }
}

/**
 * This function is used to train the perceptron (aka determine the decision boundary in the input space)
*/
void Perceptron::train(Matrix &features, Matrix &labels)
{
    //first epoch weights must be initialized.
    if (weights.size() == 0)
    {
#ifdef _DEBUG
        cout << "\nCreating weights\n";
#endif

        inputVectorSize = features.row(0).size();

        for (size_t i = 0; i <= inputVectorSize; i++)
        {
            weights.push_back(INITIAL_WEIGHTS);
        }
    }

#ifdef _DEBUG
    cout << "\nThe weight vector is size " << weights.size() << endl;
#endif

    vector<double> *featureRow;
    vector<double> targetRow;
    double target;
    //int output;
    double output;
    size_t epochsSinceLastChange = 0;
    bool weightsChanged;

#ifdef _DEBUG
    cout << "\nThere are " << features.rows() << " rows and " << features.row(0).size() << " columns (exempting bias weight) in the features matrix.\n";
    size_t output0Count = 0;
    size_t output1Count = 0;
    size_t target0Count = 0;
    size_t target1Count = 0;
#endif

    size_t epoch = 0;

    cout << "\nLearning rate: " << LEARNING_RATE << endl;

    for (; epochsSinceLastChange <= EPOCHS_SINCE_LAST_CHANGE_LIMIT && epoch < TRAINING_EPOCH_LIMIT; epoch++)
    {
        weightsChanged = false;

        features.shuffleRows(m_rand, &labels);

        //pull each row from matrix, this will be the input vector to go into the dot product function.
        for (size_t i = 0; i < features.rows(); i++)
        {
            //set up input and target data
            featureRow = &features.row(i);

            if (epoch == 0)
            {
                //add bias weight on first epoch
                (*featureRow).push_back(BIAS_WEIGHT);
            }

            targetRow = labels.row(i);
            target = targetRow.at(0);

            //uncomment for perceptron rule training
            output = processOutput(*featureRow);

            //uncomment for delta rule training
            //output = processInput(*featureRow);

#ifdef _DEBUG
            output ? ++output1Count : ++output0Count;
            target ? ++target1Count : ++target0Count;
#endif

            //uncomment for perceptron training
            //test for correctness
            if (output != target)
            {
                weightsChanged = true;
                adjustWeights(target, *featureRow, output);
            }

            //uncomment for delta training
            //adjustWeights(target, *featureRow, output);
            //weightsChanged = true;
        }

        weightsChanged ? epochsSinceLastChange = 0 : epochsSinceLastChange++;
    }

#ifdef _DEBUG
    cout << "\nTrained in " << epoch << " epochs" << (epoch == TRAINING_EPOCH_LIMIT ? " (max)\n" : "\n");
    cout << "\nFinal weights are:\n";

    for (double weight : weights)
    {
        cout << weight << ",";
    }

    cout << " (last weight is bias weight)\n";

    cout << "\nOutput " << output1Count << " 1's and " << output0Count << " 0's\n";
    cout << "Target contained " << target1Count << " 1's and " << target0Count << " 0's\n";
    cout << "Begin predictions:\n\n";
#endif
}

/**
 * This function is used to make predictions on input after training is complete.
*/
void Perceptron::predict(const vector<double> &features, vector<double> &labels)
{
    int output;

    if (features.size() == inputVectorSize)
    {
        //add bias weight since row doesn't have it
        vector<double> featuresCopy(features);
        featuresCopy.push_back(BIAS_WEIGHT);

        output = processOutput(featuresCopy);
    }
    else //row was recylced from training and already contains the bias weight
    {
        output = processOutput(features);
    }

#ifdef _DEBUG
    cout << output;
#endif

    labels.at(0) = output;
}
