#include "DTNode.h"

DTNode::DTNode(vector<tuple<vector<int>, int>> dataAndLabels)
{
    #ifdef _DEBUG
    //cout << "\nInside DTNode constructor... measuring entropy\n";
    #endif

    //measure entropy for whole set and set most common label for node in the process
    double wholeEntropy = measureEntropyAndSetCommonLabel(dataAndLabels, true);
    
    //check if stopping criteria met. If so then stop partitioning and just set most common label, initialize childNodes to nothing and set split index to -1.
    //for now we will use complete purity of subset to determine when to stop
    if(wholeEntropy == 0)
    {
        #ifdef _DEBUG
        //cout << "Data set is pure. This is a leaf node\n";
        #endif

        childNodes = map<int, unique_ptr<DTNode>>();
        //the chances of this many features being in use is small enough that this is basically to trigger exceptions if trying to go deeper in tree with no children and this wasn't caught. Better than garbage data in my opinion...
        indexToSplitOn = -1;
        isLeaf = true;
        return;
    }

    #ifdef _DEBUG
    //cout << "Entropy of whole set is " << wholeEntropy << endl;
    #endif

    double gain;
    double partitionedEntropy = 0;
    double entropyOfPartition = 0;
    int currentBestIndexForGain = -2;
    double currentBestGain = 0;

    //partition dataAndLabels into all possible subsets based on a trait
    vector<map<int, vector<tuple<vector<int>, int>>>> partitionedDataSets = vector<map<int, vector<tuple<vector<int>, int>>>>();

    int featureCount = get<0>(dataAndLabels.at(0)).size();

    if(featureCount == 0)
    {
        #ifdef _DEBUG
        //cout << "No more features to split on" << endl;
        #endif

        childNodes = map<int, unique_ptr<DTNode>>();
        indexToSplitOn = -1;
        isLeaf = true;

        return;
    }

    #ifdef _DEBUG
    //cout << "\nCreating all possible data subsets based on each one of " << featureCount << " traits...\n";
    #endif

    isLeaf = false;

    for(int featureIndex = 0; featureIndex < featureCount; featureIndex++)
    {
        partitionedDataSets.push_back(partitionData(dataAndLabels, featureIndex));
    }

    //measure the entropy of each new group of subsets

    int featureIndex = 0;

    #ifdef _DEBUG
    //cout << "Calculating possible gains on each of " << partitionedDataSets.size() << " sets (should match # traits)\n------------------------------------------------------------------------------------\n";
    #endif

    //for each possible way to partition the data...
    for(map<int, vector<tuple<vector<int>, int>>>& partitionedDataSet : partitionedDataSets)
    {
        partitionedEntropy = 0;

        //accumulate the entropy of each subset to measure overall entropy of divided set
        for(map<int, vector<tuple<vector<int>, int>>>::iterator dataSet = partitionedDataSet.begin(); dataSet != partitionedDataSet.end(); dataSet++)
        {
            double proportionOfSet = static_cast<double>((dataSet->second).size())/static_cast<double>(dataAndLabels.size());

            #ifdef _DEBUG
            //cout << "\nproportion of set = " << (dataSet->second).size() << "/" << dataAndLabels.size() << " = " << proportionOfSet;
            #endif

            entropyOfPartition = proportionOfSet * measureEntropyAndSetCommonLabel(dataSet->second, false);

            #ifdef _DEBUG
            //cout << "partitionedEntropy = " << partitionedEntropy << " + " << entropyOfPartition;
            #endif

            partitionedEntropy += entropyOfPartition;

            #ifdef _DEBUG
            //cout << " = " << partitionedEntropy << endl;
            #endif
        }

        //calculate loss of entropy (aka information gain)
        gain = wholeEntropy - partitionedEntropy;

        #ifdef _DEBUG
        //cout << "\nOverall possible gain for this data set: " << wholeEntropy << " - " << partitionedEntropy << " = " << gain << "\nfeatureIndex = " << featureIndex << "\n------------------------------------------------------------------------------------\n";
        #endif

        //track greatest gain and index that had it.
        if(gain > currentBestGain)
        {
            #ifdef _DEBUG
            //cout << "New best gain is " << gain << " at index " << featureIndex << endl;
            #endif

            currentBestGain = gain;
            currentBestIndexForGain = featureIndex;
        }

        featureIndex++;
    }

    #ifdef _DEBUG
    //cout << "Best gain: " << currentBestGain << " at feature index " << currentBestIndexForGain << endl;
    #endif

    indexToSplitOn = currentBestIndexForGain;

    //keep best partition of data
    map<int, vector<tuple<vector<int>, int>>> bestPartition = partitionedDataSets.at(currentBestIndexForGain);

    //discard inferior partitions
    partitionedDataSets.clear();

    #ifdef _DEBUG
    //cout << "Removing feature " << currentBestIndexForGain << " from data sets...\n";
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
    //cout << "Creating child nodes...\n";
    #endif

    //create child nodes for each vector
    for(map<int, vector<tuple<vector<int>, int>>>::iterator partitionIterator = bestPartition.begin(); partitionIterator != bestPartition.end(); partitionIterator++)
    {
        childNodes.emplace(partitionIterator->first, make_unique<DTNode>(partitionIterator->second));
    }

    #ifdef _DEBUG
    //cout << "Finished creating child nodes\n";
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
    //cout << "Inside labelData... label to split on is: " << indexToSplitOn << endl;
    #endif

    if(isLeaf)
    {
        #ifdef _DEBUG
        //cout << "No child nodes... returning: " << mostCommonLabel << endl << endl;
        #endif

        return mostCommonLabel;
    }

    //check split variable to determine the index of what data point to determine appropriate child node.
    //if child index corresponding to attribute value is not null, delete the index from the vector and pass the data point to the appropriate child node
    if(childNodes.count(data.at(indexToSplitOn)))
    {
        int dataValue = data.at(indexToSplitOn);

        #ifdef _DEBUG
        //cout << "Routing data on feature " << indexToSplitOn << " with value " << dataValue << endl;
        #endif

        data.erase(data.begin() + indexToSplitOn);

        return childNodes.at(dataValue)->labelData(data);
    }
    //else return most common value at this node, since there are no child nodes with that value
    else
    {
        #ifdef _DEBUG
        //cout << "No child node for data's selected feature value, returning " << mostCommonLabel << endl << endl;
        #endif

        return mostCommonLabel;
    }
}

void DTNode::printAttributeSplits(size_t depth)
{
    //on a new line, add spaces corresponding to how deep the node is in the tree
    cout << endl << string(depth, ' ');

    if(isLeaf)
    {
        //print an X to indicate this is a leaf node
        cout << "X";
    }
    else
    {
        //print the index the data was split on, followed by a colon to indicate this is not a leaf node
        cout << indexToSplitOn << ":";
    }

    cout << "\t\t\t->" << mostCommonLabel;
    //increment the depth count and recursively call this function on all child nodes.
    depth++;

    if(!isLeaf)
    {
        for(map<int, unique_ptr<DTNode>>::iterator childNodesIterator = childNodes.begin(); childNodesIterator != childNodes.end(); childNodesIterator++)
        {
            childNodesIterator->second->printAttributeSplits(depth);
        }
    }
}

void DTNode::restoreChildNodes(size_t depthToRestore)
{
    if(!depthToRestore)
    {
        return;
    }

    if(childNodes.empty())
    {
        isLeaf = true;
        return;
    }

    if(--depthToRestore)
    {
        isLeaf = false;
        
        for(map<int, unique_ptr<DTNode>>::iterator childNodesIterator = childNodes.begin(); childNodesIterator != childNodes.end(); childNodesIterator++)
        {
            childNodesIterator->second->restoreChildNodes(depthToRestore);
        }
    }
}

void DTNode::markThisAndAllChildrenAsLeaves()
{
    isLeaf = true;

    for(map<int, unique_ptr<DTNode>>::iterator childNodesIterator = childNodes.begin(); childNodesIterator != childNodes.end(); childNodesIterator++)
    {
        childNodesIterator->second->markThisAndAllChildrenAsLeaves();
    }
}

void DTNode::pruneChild(size_t depthUntilCutoff)
{
    if(!depthUntilCutoff)
    {
        return;
    }

    if(childNodes.empty())
    {
        isLeaf = true;
        return;
    }

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
        markThisAndAllChildrenAsLeaves();
    }
}

size_t DTNode::getTreeNodeCount()
{
    size_t nodeCount = 0;

    if(!isLeaf)
    {
        for(map<int, unique_ptr<DTNode>>::iterator childNodesIterator = childNodes.begin(); childNodesIterator != childNodes.end(); childNodesIterator++)
        {
            nodeCount += childNodesIterator->second->getTreeNodeCount();
        }
    }

    return ++nodeCount;
}

size_t DTNode::getTreeNodeDepth()
{
    size_t maxNodeDepth = 0;

    if(!isLeaf)
    {
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
    }
    return ++maxNodeDepth;
}

double DTNode::measureEntropyAndSetCommonLabel(vector<tuple<vector<int>, int>>& dataAndLabels, bool wholeSet)
{
    map<int, size_t> labelsToDataInstanceCounts = map<int, size_t>();

    #ifdef _DEBUG
    //cout << "\nInside measureEntropyAndSetCommonLabel\ntracking label sizes\n";
    #endif

    //track label sizes
    for(tuple<vector<int>, int>& entry : dataAndLabels)
    {
        if(!labelsToDataInstanceCounts.count(get<1>(entry)))
        {
            labelsToDataInstanceCounts.emplace(get<1>(entry), 0);    
        }
        
        labelsToDataInstanceCounts.at(get<1>(entry))++;
    }

    #ifdef _DEBUG
    //cout << "Calculating entropies...\n";
    #endif

    double entropy = 0;
    size_t greatestDataInstanceCount = 0;

    //calculate entropy and find most common label
    for(map<int, size_t>::iterator labelsToDataInstanceCountsIter = labelsToDataInstanceCounts.begin(); labelsToDataInstanceCountsIter != labelsToDataInstanceCounts.end(); labelsToDataInstanceCountsIter++)
    {
        double p = (static_cast<double>(labelsToDataInstanceCountsIter->second))/static_cast<double>(dataAndLabels.size());
        #ifdef _DEBUG
        //cout << "p: " << labelsToDataInstanceCountsIter->second << "/" << dataAndLabels.size() << " = " << p << ", ";
        //cout.flush();
        #endif

        double log2p = log2(p);
        #ifdef _DEBUG
        //cout << "log2p: " << log2p << ", ";
        //cout.flush();
        #endif

        double subsetEntropy = (p * log2p);
        #ifdef _DEBUG
        //cout << "subset entropy: " << subsetEntropy << endl;
        #endif

        entropy -= subsetEntropy;

        if(wholeSet && labelsToDataInstanceCountsIter->second > greatestDataInstanceCount)
        {
            mostCommonLabel = labelsToDataInstanceCountsIter->first;
            greatestDataInstanceCount = labelsToDataInstanceCountsIter->second;
        }
    }

    #ifdef _DEBUG
    //cout << "Total entropy for this set: " << entropy << endl;
    #endif

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