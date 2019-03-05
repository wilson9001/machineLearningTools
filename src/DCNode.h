#ifndef DCNODE_H
#define DCNODE_H

#include <vector>
#include <memory>
#include <tuple>
#include <map>
#include <math.h>
#include <iostream>

using namespace std;

class DCNode
{
    private:
    vector<unique_ptr<DCNode>> childNodes;
    size_t indexToSplitOn;
    bool stoppingCriteriaMet(vector<tuple<vector<int>, int>> dataAndLabels);

    protected:
    int mostCommonLabel;
    //if wholeSet is true then the node's most common label value will be set.
    double measureEntropyAndSetCommonLabel(vector<tuple<vector<int>, int>>& dataAndLabels, bool wholeSet);

    public:
    const int PRUNEALLCHILDNODES = -1;
    
    DCNode();
    DCNode(vector<tuple<vector<int>, int>> dataAndLabels);
    ~DCNode();

    vector<unique_ptr<DCNode>>& getChildNodes();
    
    virtual int labelData(vector<int> data);
    virtual void printAttributeSplits(size_t depth);
    //pass in vector with PRUNEALLCHILDNODES as first element to prune all child nodes.
    virtual void pruneChild(size_t depthUntilCutoff);
    virtual size_t getTreeNodeCount();
    virtual size_t getTreeNodeDepth();
};

#endif