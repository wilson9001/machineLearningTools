#include "perceptron.h"
#include "error.h"
#include <string>

using namespace std;

/**
 * This helper function takes the net output from processing inputs and then converts it to a 0 or 1, since this is a perceptron.
*/
int Perceptron::processOutput(vector<double> &inputs)
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

        for (size_t i = 0; i < inputVectorSize; i++)
        {
            total += inputs[i] * weights[i];
        }

        return total;
    }
    else
    {
        ThrowError("features vector size != weight vector size! ", to_string(inputs.size()),", ", to_string(weights.size()));
        return 0;
    }
}

/**
 * This helper function is used to avoid code duplication in processing input.
*/
double Perceptron::processInput(vector<double> &inputs)
{
    //add bias
    //inputs.push_back(BIAS_WEIGHT);

    //perform input processing
    double net = dotProduct(inputs);

    //remove bias
    //inputs.pop_back();

    return net;
}

/**
 * Adjust weights by subtractig the target from the output, multiply by the learning rate and the existing weight value.
*/
void Perceptron::adjustWeights(double target, double output)
{
    for (size_t i = 0; i <= inputVectorSize; i++)
    {
        weights.at(i) += ((target - output) * LEARNING_RATE * weights.at(i));
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
        inputVectorSize = features.row(0).size();

        for (size_t i = 0; i <= inputVectorSize; i++)
        {
            weights.push_back(INITIAL_WEIGHTS);
        }
    }

    vector<double> featureRow, targetRow;
    double target;
    int output;
    bool weightsChanged;

    for (size_t epoch = 0; epoch < TRAINING_EPOCH_LIMIT; epoch++)
    {
        weightsChanged = false;

        features.shuffleRows(m_rand, &labels);

        //pull each row from matrix, this will be the input vector to go into the dot product function.
        for (size_t i = 0; i < features.rows(); i++)
        {
            //set up input and target data
            featureRow = features.row(i);

            if (epoch == 0)
            {
                //add bias weight on first epoch
                featureRow.push_back(BIAS_WEIGHT);
            }

            targetRow = labels.row(i);
            target = targetRow.at(0);

            output = processOutput(featureRow);

            //test for correctness
            if (output != target)
            {
                weightsChanged = true;
                adjustWeights(target, output);
            }
        }

        if (!weightsChanged)
        {
            break;
        }
    }
}

/**
 * This function is used to make predictions on input after training is complete.
*/
void Perceptron::predict(const vector<double> &features, vector<double> &labels)
{
    vector<double> featuresCopy(features);

    labels.push_back(processOutput(featuresCopy));
}
