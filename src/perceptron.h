#ifndef PERCEPTRON_H
#define PERCEPTRON_H

#include "learner.h"
#include "rand.h"
#include <vector>

using namespace std;

class Perceptron : public SupervisedLearner
{
  private:
    /**
     * This helper function takes the net output from processing inputs and then converts it to a 0 or 1, since this is a perceptron.
    */
    int processOutput(vector<double> &inputs);

  protected:
    Rand &m_rand;
    vector<double> m_labelVec;
    double dotProduct(vector<double> features, vector<double> weights);
    vector<double> weights;
    size_t inputVectorSize;
    const double BIAS_WEIGHT = 0;
    const int THRESHOLD = 0;
    const int INITIAL_WEIGHTS = 0;
    const double LEARNING_RATE = .1;
    const size_t TRAINING_EPOCH_LIMIT = 10000;

    /**
     * This helper function is used to process inputs to the perceptron by performing dot products.
    */
    double dotProduct(const vector<double> &inputs);

    /**
     * This helper function is used to avoid code duplication in processing perceptron input.
    */
    double processInput(vector<double> &inputs);

    void adjustWeights(double target, double output);

  public:
    Perceptron(Rand &r) : SupervisedLearner(), m_rand(r)
    {
        inputVectorSize = 0;
    }

    virtual ~Perceptron()
    {
    }

    // Train the model to predict the labels
    virtual void train(Matrix &features, Matrix &labels);

    // Evaluate the features and predict the labels
    virtual void predict(const vector<double> &features, vector<double> &labels);
};

#endif