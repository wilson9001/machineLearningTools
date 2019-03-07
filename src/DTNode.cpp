#include "DTNode.h"

DTNode::DTNode(vector<tuple<vector<int>, int>> dataAndLabels)
{
    #ifdef _DEBUG
    cout << "Inside DTNode constructor... measuring entropy\n";
    #endif

    //measure entropy for whole set and set most common label for node in the process
    double wholeEntropy = measureEntropyAndSetCommonLabel(dataAndLabels, true);
    
    //check if stopping criteria met. If so then stop partitioning and just set most common label, initialize childNodes to nothing and set split index to -1.
    //for now we will use complete purity of subset to determine when to stop
    if(!wholeEntropy)
    {
        #ifdef _DEBUG
        cout << "Data set is pure. This is a leaf node\n";
        #endif

        childNodes = map<int, unique_ptr<DTNode>>();
        //the chances of this many features being in use is small enough that this is basically to trigger exceptions if trying to go deeper in tree with no children and this wasn't caught. Better than garbage data in my opinion...
        indexToSplitOn = -1;
        return;
    }

    #ifdef _DEBUG
    cout << "Entropy of whole set is " << wholeEntropy << endl;
    #endif

    double gain;
    int currentBestIndexForGain;
    double currentBestGain = 0;

    //partition dataAndLabels into all possible subsets based on a trait
    vector<map<int, vector<tuple<vector<int>, int>>>> partitionedDataSets = vector<map<int, vector<tuple<vector<int>, int>>>>();

    int featureCount = get<0>(dataAndLabels.at(0)).size();

    #ifdef _DEBUG
    cout << "Creating all possible data subsets based on one trait...\n";
    #endif

    for(int featureIndex = 0; featureIndex < featureCount; featureIndex++)
    {
        partitionedDataSets.push_back(partitionData(dataAndLabels, featureIndex));
    }

    //measure the entropy of each new group of subsets

    int featureIndex = 0;

    #ifdef _DEBUG
    cout << "Calculating possible gains...\n";
    #endif

    for(map<int, vector<tuple<vector<int>, int>>>& partitionedDataSet : partitionedDataSets)
    {
        gain = wholeEntropy;

        //from current entropy subtract the entropy of each subset in group to measure overall information gain of partitioning the data this way
        for(map<int, vector<tuple<vector<int>, int>>>::iterator dataSet = partitionedDataSet.begin(); dataSet != partitionedDataSet.end(); dataSet++)
        {
            gain -= measureEntropyAndSetCommonLabel(dataSet->second, false);
        }

        #ifdef _DEBUG
        cout << gain << endl;
        #endif

        //track greatest gain and index that had it.
        if(gain > currentBestGain)
        {
            currentBestGain = gain;
            currentBestIndexForGain = featureIndex;
        }

        featureIndex++;
    }

    #ifdef _DEBUG
    cout << "Best gain: " << currentBestGain << " at feature index " << currentBestIndexForGain << endl;
    #endif

    //keep best partition of data
    map<int, vector<tuple<vector<int>, int>>> bestPartition = partitionedDataSets.at(currentBestIndexForGain);

    //discard inferior partitions
    partitionedDataSets.clear();

    #ifdef _DEBUG
    cout << "Removing feature from data sets...\n";
    #endif

    //remove feature to split on from data set
    for(map<int, vector<tuple<vector<int>, int>>>::iterator subsetIterator = bestPartition.begin(); subsetIterator != bestPartition.end(); subsetIterator++)
    {
        for(tuple<vector<int>, int>& dataRow : subsetIterator->second)
        {
            get<0>(dataRow).erase((get<0>(dataRow)).begin() + currentBestIndexForGain);
        }
    }

    #ifdef _DEBUG
    cout << "Creating child nodes...\n";
    #endif

    //create child nodes for each vector
    for(map<int, vector<tuple<vector<int>, int>>>::iterator partitionIterator = bestPartition.begin(); partitionIterator != bestPartition.end(); partitionIterator++)
    {
        childNodes.emplace(partitionIterator->first, make_unique<DTNode>(partitionIterator->second));
    }

    #ifdef _DEBUG
    cout << "Finished creating child nodes\n";
    #endif
}

DTNode::~DTNode()
{}

map<int, unique_ptr<DTNode>>& DTNode::getChildNodes()
{
    return ref(childNodes);
}

int DTNode::labelData(vector<int> data)
{
    #ifdef _DEBUG
    cout << "Inside labelData...\n";
    #endif

    //check split variable to determine the index of what data point to determine appropriate child node.
    //if child index corresponding to attribute value is not null, delete the index from the vector and pass the data point to the appropriate child node
    if(childNodes.count(data.at(indexToSplitOn)))
    {
        #ifdef _DEBUG
        cout << "Routing data on feature " << indexToSplitOn << endl;
        #endif

        int dataValue = data.at(indexToSplitOn);
        data.erase(data.begin() + indexToSplitOn);

        return childNodes.at(dataValue)->labelData(data);
    }
    //else return most common value at this node, since there are no child nodes with that value
    else
    {
        #ifdef _DEBUG
        cout << "No child node for data's selected feature value, returning " << mostCommonLabel << endl;
        #endif

        return mostCommonLabel;
    }
    
}

void DTNode::printAttributeSplits(size_t depth)
{
    //on a new line, add spaces corresponding to how deep the node is in the tree
    cout << endl << string(depth, ' ');

    if(childNodes.empty())
    {
        //print an X to indicate this is a leaf node
        cout << "X";
    }
    else
    {
        //print the index the data was split on, followed by a colon to indicate this is not a leaf node
        cout << indexToSplitOn << ":";
    }

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