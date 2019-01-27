#include "perceptron.h"
#include "error.h"
#include <string>

using namespace std;

class Perceptron : public SupervisedLearner
{
  private:
    /**
     * This helper function takes the net output from processing inputs and then converts it to a 0 or 1, since this is a perceptron.
    */
    int processOutput(vector<double> &inputs)
    {
        double net = processInput(inputs);

        //determine output
        return net > THRESHOLD;
    }

  protected:
    Rand &m_rand;
    vector<double> m_labelVec;
    vector<double> weights;
    size_t inputVectorSize;
    const double BIAS_WEIGHT = 0;
    const int THRESHOLD = 0;
    const int INITIAL_WEIGHTS = 0;
    const double LEARNING_RATE = .1;
    const int THRESHOLD = 0;

    /**
     * This helper function is used to process inputs to the perceptron by performing dot products.
    */
    double dotProduct(const vector<double> &inputs, const vector<double> &weights)
    {
        if (inputs.size() == weights.size())
        {
            double total = 0;

            for (size_t i = 0; i < inputs.size(); i++)
            {
                total += inputs[i] * weights[i];
            }

            return total;
        }
        else
        {
            ThrowError("features vector size != weight vector size!", to_string(inputs.size()), to_string(weights.size()));
        }
    }

    /**
     * This helper function is used to avoid code duplication in processing input.
    */
    double processInput(vector<double> &inputs)
    {
        //add bias
        inputs.push_back(BIAS_WEIGHT);

        //perform input processing
        double net = dotProduct(inputs, weights);

        //remove bias
        inputs.pop_back();

        return net;
    }

    void adjustWeights(double target, double output)
    {
        //adjust weights by subtractig the target from the output, multiply by the learning rate and the existing weight value.
        for (size_t i = 0; i < inputVectorSize; i++)
        {
            weights.at(i) += ((target - output) * LEARNING_RATE * weights.at(i));
        }
    }

  public:
  /**
   * This function is used to train the perceptron (aka determine the decision boundary in the input space)
  */
    virtual void train(Matrix &features, Matrix &labels, size_t index)
    {
        //first epoch weights must be initialized.
        if (weights.size() == 0)
        {
            inputVectorSize = features.row(0).size();

            for (int i = 0; i <= inputVectorSize; i++)
            {
                weights.push_back(INITIAL_WEIGHTS);
            }
        }

        vector<double> featureRow, targetRow;
        double target;
        int output;

        //pull each row from matrix, this will be the input vector to go into the dot product function.
        for (size_t i = 0; i < features.rows(); i++)
        {
            //set up input and target data
            featureRow = features.row(i);
            targetRow = labels.row(i);
            target = targetRow.at(0);

            output = processOutput(featureRow);

            //test for correctness
            if (output != target)
            {
                adjustWeights(target, output);
            }
        }
    }

    /**
     * This function is used to make predictions on input after training is complete.
    */
    virtual void predict(const vector<double> &features, vector<double> &labels)
    {
        vector<double> featuresCopy(features);

        labels.at(0) = processOutput(featuresCopy);
    }
};
