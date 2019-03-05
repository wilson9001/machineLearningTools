#include "DCNode.h"

DCNode::DCNode(): childNodes(vector<unique_ptr<DCNode>>()), indexToSplitOn(-1)
{}

DCNode::DCNode(vector<tuple<vector<int>, int>> dataAndLabels)
{
    //measure entropy for whole set and set most common label for node in the process
    double wholeEntropy = measureEntropyAndSetCommonLabel(dataAndLabels, true);

    size_t currentBestIndexForGain;
    double currentBestGain = 0;

    //map dataAndLabels into tuple vectors based on each trait and measure the entropy of each new group. Aggregate them and subtract from current entropy to measure information gain.

    //track greatest gain and index that had it.

    //set index to divide on

    //create vector with index for all known possible options based on highest value

    //partition data into vectors for each option

    //create child nodes for each vector, checking if it should be a leaf node or not by calling stoppingCriteria function
}

DCNode::~DCNode()
{
    childNodes.clear();
}

vector<unique_ptr<DCNode>>& DCNode::getChildNodes()
{
    return ref(childNodes);
}

int DCNode::labelData(vector<int> data)
{
    //check split variable to determine the index of what data point to determine appropriate child node.

    //if child index corresponding to attribute value is not null, delete the index from the vector and pass the data point to the appropriate child node

    //else return most common value at this node, since there are no child nodes with that value
}

void DCNode::printAttributeSplits(size_t depth)
{
    //on a new line, add spaces corresponding to how deep the node is in the tree, then print the index the data was split on followed by a colon to indicate this is not a leaf node
    cout << endl << string(depth, ' ') << indexToSplitOn << ":";

    //increment the depth count and recursively call this function on all child nodes.
    depth++;

    for(unique_ptr<DCNode>& child : childNodes)
    {
        child->printAttributeSplits(depth);
    }
}

void DCNode::pruneChild(size_t depthUntilCutoff)
{
    //if depth is > 0, pass down to children
    if(--depthUntilCutoff)
    {
        for(unique_ptr<DCNode>& child : childNodes)
        {
            child->pruneChild(depthUntilCutoff);
        }
    }
    else
    {
        //otherwise remove child nodes
        childNodes.clear();
    }
    
}

size_t DCNode::getTreeNodeCount()
{
    size_t nodeCount = 0;

    for(unique_ptr<DCNode>& childNode : childNodes)
    {
        nodeCount += childNode->getTreeNodeCount();
    }

    return ++nodeCount;
}

size_t DCNode::getTreeNodeDepth()
{
    size_t maxNodeDepth = 0;

    vector<size_t> possibleDepths = vector<size_t>();
    possibleDepths.reserve(childNodes.size());

    for(unique_ptr<DCNode>& childNode : childNodes)
    {
        possibleDepths.push_back(childNode->getTreeNodeDepth());
    }

    for(size_t depth : possibleDepths)
    {
        if(depth > maxNodeDepth)
        {
            maxNodeDepth = depth;
        }
    }

    return ++maxNodeDepth;
}

double DCNode::measureEntropyAndSetCommonLabel(vector<tuple<vector<int>, int>>& dataAndLabels, bool wholeSet)
{
    map<int, size_t> labelsToFeatureCount = map<int, size_t>();
    double entropy = 0;
    size_t greatestLabelCount = 0;

    //track label sizes
    for(tuple<vector<int>, int>& entry : dataAndLabels)
    {
        if(labelsToFeatureCount.count(get<1>(entry)))
        {
            labelsToFeatureCount.at(get<1>(entry))++;
        }
        else
        {
            labelsToFeatureCount.emplace(get<1>(entry), 1);
        }
    }

    //calculate entropy and find most common label
    for(map<int, size_t>::iterator iter = labelsToFeatureCount.begin(); iter != labelsToFeatureCount.end(); iter++)
    {
        entropy -= (((iter->second)/dataAndLabels.size()) * log2((iter->second)/dataAndLabels.size()));

        if(wholeSet && iter->second > greatestLabelCount)
        {
            mostCommonLabel = iter->first;
            greatestLabelCount = iter->second;
        }
    }

    return entropy;
}