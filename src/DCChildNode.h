#ifndef DCCHILDNODE_H
#define DCCHILDNODE_H

#include "DCNode.h"
#include <vector>

class DCChildNode : public DCNode
{   
    public:
    DCChildNode(vector<tuple<vector<int>, int>> dataAndLabels);
    ~DCChildNode();

    int labelData(vector<double> data);
    void pruneChild(size_t depthUntilCutoff);
    void printAttributeSplits(size_t depth);
    size_t getChildNodeCount();
    size_t getTreeNodeDepth();
};

#endif