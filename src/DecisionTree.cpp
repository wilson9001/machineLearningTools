#include "DecisionTree.h"

DecisionTree::DecisionTree(Rand &r): SupervisedLearner(), m_rand(r)
{
    root = nullptr;
}

DecisionTree::~DecisionTree()
{}

size_t DecisionTree::getNodeCount()
{
    //call root's getNodeCount to initiate resursive tally
    return root->getTreeNodeCount();
}

size_t DecisionTree::getTreeDepth()
{
    //call root's getTreeDepth to initiate resursive search
    return root->getTreeNodeDepth();

    cout << endl;
}

void DecisionTree::createTree(vector<tuple<vector<int>, int>> dataAndLabel)
{
    if(dataAndLabel.empty())
    {
        ThrowError("Attempted to create decision tree on empty data set");
    }

    //call root's createTree function to begin recursively creating tree
    root = make_unique<DTNode>(dataAndLabel);
}

int DecisionTree::classifyData(vector<int> data)
{
    //call root's recursive labelData function to begin resursive sort of data into tree
    if(root)
    {
        return root->labelData(data);
    }
    else
    {
        ThrowError("Attempted to classify data with uninitialized tree");
        //this will never be reached but the compiler issues a warning without it.
        return -1;
    }
}

void DecisionTree::printAttributeSplits()
{
    //call root's printAttributeSplits function with a value of 0 to start with no spaces
    root->printAttributeSplits(PRINTTREEMARGIN);
}

void DecisionTree::pruneTree(size_t depthLimit)
{
    //call root's pruneChild function with the depth limit passed in so the root can begin recursively removing child nodes who are too deep.
    root->pruneChild(depthLimit);
}

void DecisionTree::train(Matrix &features, Matrix &labels)
{
    //conditionally create validation set

    //pack features and labels into a vector of tuples to simplify data handling in the tree
    vector<tuple<vector<int>, int>> dataAndLabels = vector<tuple<vector<int>, int>>();

    vector<int> castFeatureRow = vector<int>();
    int label;

    for(size_t i = 0; i < features.rows(); i++)
    {
        vector<double>& featureRow = features.row(i);
        vector<double>& labelRow = labels.row(i);

        castFeatureRow.clear();

        //cast features
        for(double feature : featureRow)
        {
            castFeatureRow.push_back(static_cast<int>(feature));
        }

        //cast label
        label = static_cast<int>(labelRow.at(0));

        //pack it all together
        dataAndLabels.push_back(make_tuple(castFeatureRow, label));
    }

    //call the create Tree function to recursively create the tree
    createTree(dataAndLabels);

    //conditionally print the new tree's structure
    if(getenv("TREE") && !strncmp(getenv("TREE"), "y", 1))
    {
        printAttributeSplits();
    }

    //measure training accuracy and possibly validation set accuracy
    double trainingMeanSquaredError = pow(measureAccuracy(features, labels), 2);

    cout << "Training Mean Squared Error: " << trainingMeanSquaredError << endl;
}

void DecisionTree::predict(const vector<double> &features, vector<double> &labels)
{
    //copy features so they can be removed in decision tree
    vector<int> featuresCopy = vector<int>();

    for(double feature : features)
    {
        featuresCopy.push_back(static_cast<int>(feature));
    }

    //call the classifyData function to recursively label the data.
    double label = root->labelData(featuresCopy);

    //set value in label with returned answer
    labels.at(0) = static_cast<double>(label);
}