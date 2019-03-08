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
    #ifdef _DEBUG
    //cout << "Inside createTree...\n";
    #endif

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
    #ifdef _DEBUG
    cout << "Beginning training...\n";
    #endif
    //conditionally create validation set

    //pack features and labels into a vector of tuples to simplify data handling in the tree
    vector<tuple<vector<int>, int>> dataAndLabels = vector<tuple<vector<int>, int>>();

    vector<int> castFeatureRow = vector<int>();
    int label;

    #ifdef _DEBUG
    cout << "Packing data and labels into tuples...\n";
    #endif

    for(size_t i = 0; i < features.rows(); i++)
    {
        vector<double>& featureRow = features.row(i);
        vector<double>& labelRow = labels.row(i);
        double safeFeature;

        castFeatureRow.clear();

        #ifdef _DEBUG
        //cout << "Casting features:\n";
        #endif

        //cast features
        for(double feature : featureRow)
        {
            #ifdef _DEBUG
            //cout << feature << " ";
            //cout.flush();
            #endif
            
            safeFeature = feature < 0 ? UNKNOWNCATEGORYVALUE : feature;

            castFeatureRow.push_back(static_cast<int>(safeFeature));
        }

        #ifdef _DEBUG
        //cout << "\nCasting label: " << labelRow.at(0) << endl;
        #endif

        //cast label
        label = static_cast<int>(labelRow.at(0));

        #ifdef _DEBUG
        //cout << "Packing it all together...\n";
        #endif
        //pack it all together
        dataAndLabels.push_back(make_tuple(castFeatureRow, label));
    }

    //call the create Tree function to recursively create the tree
    createTree(dataAndLabels);

    //conditionally print the new tree's structure
    if(getenv("TREE") && !strncmp(getenv("TREE"), "y", 1))
    {
        printAttributeSplits();
        cout << endl << endl;
    }

    //measure training accuracy and possibly validation set accuracy
    double trainingMeanSquaredError = pow(measureAccuracy(features, labels), 2);

    cout << "Training Mean Squared Error: " << trainingMeanSquaredError << endl;
}

void DecisionTree::predict(const vector<double> &features, vector<double> &labels)
{
    #ifdef _DEBUG
    //cout << "Inside predict function" << endl;
    #endif

    //copy features so they can be removed in decision tree
    vector<int> featuresCopy = vector<int>();

    double safeFeature;

    for(double feature : features)
    {
        safeFeature = feature < 0 ? -1 : feature;
        featuresCopy.push_back(static_cast<int>(safeFeature));
    }

    //call the classifyData function to recursively label the data.
    int label = root->labelData(featuresCopy);

    #ifdef _DEBUG
    //cout << "Raw label value is: " << label << ", cast label is: ";
    //cout.flush();
    #endif

    double castLabel = static_cast<double>(label);

    #ifdef _DEBUG
    //cout << castLabel << endl << endl;
    #endif

    //set value in label with returned answer
    labels.at(0) = castLabel;
}