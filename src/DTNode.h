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
    bool isLeaf;
    void markThisAndAllChildrenAsLeaves();
    size_t acceptableDataCountFloor;

    protected:
    int mostCommonLabel;

    //if wholeSet is true then the node's most common label value will be set.
    double measureEntropyAndSetCommonLabel(vector<tuple<vector<int>, int>>& dataAndLabels, bool wholeSet);
    //pass in an index < 0 to partition by label.
    map<int, vector<tuple<vector<int>, int>>> partitionData(vector<tuple<vector<int>, int>> dataAndLabels, int index);
    
    public:
    DTNode(vector<tuple<vector<int>, int>> dataAndLabels, size_t acceptableDataCountFloor);
    ~DTNode();

    map<int, unique_ptr<DTNode>>& getChildNodes();
    
    int labelData(vector<int> data);
    void printAttributeSplits(size_t depth);
    //passing in 0 does nothing, passing in 1 leaves only root node
    void pruneChild(size_t depthUntilCutoff);
    //passing in 0 does nothing, passing in 1 leaves only root node. Passing in 2 leaves root node and root's children
    void restoreChildNodes(size_t depthToRestore);
    size_t getTreeNodeCount();
    size_t getTreeNodeDepth();
};

#endif