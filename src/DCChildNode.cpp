#include "DCChildNode.h"

DCChildNode::DCChildNode(vector<tuple<vector<int>, int>> dataAndLabels)
{
    //set most common label for remaining data
    measureEntropyAndSetCommonLabel(dataAndLabels, true);
}

DCChildNode::~DCChildNode()
{}

int DCChildNode::labelData(vector<double> data)
{
    //we have reached the end of this line, so we must predict
    return mostCommonLabel;
}

void DCChildNode::pruneChild(size_t depthUntilCutoff)
{
    //this offshoot didn't go deep enough to require pruning
    return;
}

void DCChildNode::printAttributeSplits(size_t depth)
{
    //on a new line, add spaces corresponding to how deep the tree is, then print an X to signify this is a leaf node
    cout << endl << string(depth, ' ') << "X";
}

size_t DCChildNode::getChildNodeCount()
{
    return 1;
}

size_t DCChildNode::getTreeNodeDepth()
{
    return 1;
}