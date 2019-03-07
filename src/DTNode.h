#ifndef DTNODE_H
#define DTNODE_H

#include <vector>
#include <memory>
#include <tuple>
#include <map>
#include <math.h>
#include <iostream>

using namespace std;

class DTNode
{
    private:
    map<int, unique_ptr<DTNode>> childNodes;
    size_t indexToSplitOn;
    bool stoppingCriteriaMet(vector<tuple<vector<int>, int>> dataAndLabels);

    protected:
    int mostCommonLabel;

    //if wholeSet is true then the node's most common label value will be set.
    double measureEntropyAndSetCommonLabel(vector<tuple<vector<int>, int>>& dataAndLabels, bool wholeSet);
    //pass in an index < 0 to partition by label.
    map<int, vector<tuple<vector<int>, int>>> partitionData(vector<tuple<vector<int>, int>> dataAndLabels, int index);
    
    public:
    const int PRUNEALLCHILDNODES = -1;
    
    DTNode(vector<tuple<vector<int>, int>> dataAndLabels);
    ~DTNode();

    map<int, unique_ptr<DTNode>>& getChildNodes();
    
    virtual int labelData(vector<int> data);
    virtual void printAttributeSplits(size_t depth);
    //pass in vector with PRUNEALLCHILDNODES as first element to prune all child nodes.
    virtual void pruneChild(size_t depthUntilCutoff);
    virtual size_t getTreeNodeCount();
    virtual size_t getTreeNodeDepth();
};

#endif