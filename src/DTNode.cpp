#include "DTNode.h"

DTNode::DTNode(vector<tuple<vector<int>, int>> dataAndLabels)
{
    //measure entropy for whole set and set most common label for node in the process
    double wholeEntropy = measureEntropyAndSetCommonLabel(dataAndLabels, true);
    
    //check if stopping criteria met. If so then stop partitioning and just set most common label, initialize childNodes to nothing and set split index to -1.
    if(!wholeEntropy)
    {
        //for now we will use complete purity of subset to determine when to stop
        childNodes = map<int, unique_ptr<DTNode>>();
        //the chances of this many features being in use is small enough that this is basically to trigger exceptions if trying to go deeper in tree with no children and this wasn't caught. Better than garbage data in my opinion...
        indexToSplitOn = -1;
        return;
    }

    double gain;
    int currentBestIndexForGain;
    double currentBestGain = 0;

    
    //partition dataAndLabels into all possible subsets based on a trait
    vector<map<int, vector<tuple<vector<int>, int>>>> partitionedDataSets = vector<map<int, vector<tuple<vector<int>, int>>>>();

    int featureCount = get<0>(dataAndLabels.at(0)).size();

    for(int featureIndex = 0; featureIndex < featureCount; featureIndex++)
    {
        partitionedDataSets.push_back(partitionData(dataAndLabels, featureIndex));
    }

    //measure the entropy of each new group of subsets

    int featureIndex = 0;

    for(map<int, vector<tuple<vector<int>, int>>>& partitionedDataSet : partitionedDataSets)
    {
        gain = wholeEntropy;

        //from current entropy subtract the entropy of each subset in group to measure overall information gain of partitioning the data this way
        for(map<int, vector<tuple<vector<int>, int>>>::iterator dataSet = partitionedDataSet.begin(); dataSet != partitionedDataSet.end(); dataSet++)
        {
            gain -= measureEntropyAndSetCommonLabel(dataSet->second, false);
        }

        //track greatest gain and index that had it.
        if(gain > currentBestGain)
        {
            currentBestGain = gain;
            currentBestIndexForGain = featureIndex;
        }

        featureIndex++;
    }

    //keep best partition of data
    map<int, vector<tuple<vector<int>, int>>> bestPartition = partitionedDataSets.at(currentBestIndexForGain);

    //discard inferior partitions
    partitionedDataSets.clear();

    //remove feature to split on from data set
    for(map<int, vector<tuple<vector<int>, int>>>::iterator subsetIterator = bestPartition.begin(); subsetIterator != bestPartition.end(); subsetIterator++)
    {
        for(tuple<vector<int>, int>& dataRow : subsetIterator->second)
        {
            get<0>(dataRow).erase((get<0>(dataRow)).begin() + currentBestIndexForGain);
        }
    }

    //create child nodes for each vector
    for(map<int, vector<tuple<vector<int>, int>>>::iterator partitionIterator = bestPartition.begin(); partitionIterator != bestPartition.end(); partitionIterator++)
    {
        childNodes.emplace(partitionIterator->first, make_unique<DTNode>(partitionIterator->second));
    }
}

DTNode::~DTNode()
{}

map<int, unique_ptr<DTNode>>& DTNode::getChildNodes()
{
    return ref(childNodes);
}

int DTNode::labelData(vector<int> data)
{
    //check if there are child nodes. If not then return most common label
    if(childNodes.empty())
    {
        return mostCommonLabel;
    }

    //check split variable to determine the index of what data point to determine appropriate child node.
    //if child index corresponding to attribute value is not null, delete the index from the vector and pass the data point to the appropriate child node
    if(childNodes.count(data.at(indexToSplitOn)))
    {
        int dataValue = data.at(indexToSplitOn);
        data.erase(data.begin() + indexToSplitOn);

        return childNodes.at(dataValue)->labelData;
    }
    //else return most common value at this node, since there are no child nodes with that value
    else
    {
        return mostCommonLabel;
    }
    
}

void DTNode::printAttributeSplits(size_t depth)
{
    //on a new line, add spaces corresponding to how deep the node is in the tree
    //if the node has children then print the index the data was split on, followed by a colon to indicate this is not a leaf node
    //TODO: otherwise print an X to indicate this is a leaf node

    cout << endl << string(depth, ' ') << indexToSplitOn << ":";

    //increment the depth count and recursively call this function on all child nodes.
    depth++;

    for(map<int, unique_ptr<DTNode>>::iterator childNodesIterator = childNodes.begin(); childNodesIterator != childNodes.end(); childNodesIterator++)
    {
        childNodesIterator->second->printAttributeSplits(depth);
    }
}

void DTNode::pruneChild(size_t depthUntilCutoff)
{
    //if depth is > 0, pass down to children
    if(--depthUntilCutoff)
    {
        for(map<int, unique_ptr<DTNode>>::iterator childNodesIterator = childNodes.begin(); childNodesIterator != childNodes.end(); childNodesIterator++)
        {
            childNodesIterator->second->pruneChild(depthUntilCutoff);
        }
    }
    else
    {
        //otherwise remove child nodes
        childNodes.clear();
    }
}

size_t DTNode::getTreeNodeCount()
{
    size_t nodeCount = 0;

    for(map<int, unique_ptr<DTNode>>::iterator childNodesIterator = childNodes.begin(); childNodesIterator != childNodes.end(); childNodesIterator++)
    {
        nodeCount += childNodesIterator->second->getTreeNodeCount();
    }

    return ++nodeCount;
}

size_t DTNode::getTreeNodeDepth()
{
    size_t maxNodeDepth = 0;

    vector<size_t> possibleDepths = vector<size_t>();
    possibleDepths.reserve(childNodes.size());

    for(map<int, unique_ptr<DTNode>>::iterator childNodesIterator = childNodes.begin(); childNodesIterator != childNodes.end(); childNodesIterator++)
    {
        possibleDepths.push_back(childNodesIterator->second->getTreeNodeDepth());
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

double DTNode::measureEntropyAndSetCommonLabel(vector<tuple<vector<int>, int>>& dataAndLabels, bool wholeSet)
{
    map<int, size_t> labelsToDataInstanceCounts = map<int, size_t>();

    //track label sizes
    for(tuple<vector<int>, int>& entry : dataAndLabels)
    {
        if(!labelsToDataInstanceCounts.count(get<1>(entry)))
        {
            labelsToDataInstanceCounts.emplace(get<1>(entry), 0);    
        }
        
        labelsToDataInstanceCounts.at(get<1>(entry))++;
    }

    double entropy = 0;
    size_t greatestDataInstanceCount = 0;

    //calculate entropy and find most common label
    for(map<int, size_t>::iterator labelsToDataInstanceCountsIter = labelsToDataInstanceCounts.begin(); labelsToDataInstanceCountsIter != labelsToDataInstanceCounts.end(); labelsToDataInstanceCountsIter++)
    {
        entropy -= (((labelsToDataInstanceCountsIter->second)/dataAndLabels.size()) * log2((labelsToDataInstanceCountsIter->second)/dataAndLabels.size()));

        if(wholeSet && labelsToDataInstanceCountsIter->second > greatestDataInstanceCount)
        {
            mostCommonLabel = labelsToDataInstanceCountsIter->first;
            greatestDataInstanceCount = labelsToDataInstanceCountsIter->second;
        }
    }

    return entropy;
}

map<int, vector<tuple<vector<int>, int>>> DTNode::partitionData(vector<tuple<vector<int>, int>> dataAndLabels, int index)
{
    map<int, vector<tuple<vector<int>, int>>> partitionedDataAndLabels = map<int, vector<tuple<vector<int>, int>>>();

    for(tuple<vector<int>, int> dataRow : dataAndLabels)
    {
        int key = index >= 0 ? (get<0>(dataRow)).at(index) : get<1>(dataRow);

        if(!partitionedDataAndLabels.count(key))
        {
            partitionedDataAndLabels.emplace(key, vector<tuple<vector<int>, int>>());
        }

        partitionedDataAndLabels.at(key).push_back(dataRow);
    }

    return partitionedDataAndLabels;
}