#ifndef NODE_H
#define NODE_H

#include <string>
#include <fstream>
#include <vector>
#include <memory>
#include "error.h"
#ifdef _DEBUG
#include <iostream>
#endif
//#include "MiddleNode.h"
//#include "NonInputNode.h"

using namespace std;

class Node
{
    private:
    //Access Linux kernel's uuid generator.
    const string KERNELUUIDGEN = "/proc/sys/kernel/random/uuid";

    protected:
    double output;
    string uuid;

    public:
    Node();
    Node(double initialOutput);
    virtual ~Node();
    double getOutput();
    void setOutput(double newOutput);
    string getUUID();

    //This is kind of hackish but it simplifies higher-level operations
    
    virtual void calculateError(double target)
    {
        #ifdef _DEBUG
        cout << "Calculate error called in Node class. Doing nothing.\n";
        #endif
    }
    virtual vector<double> getWeights(){return vector<double>();}
    virtual void setWeights(vector<double> newWeights){}
    virtual double getError(){return 0;}
    virtual vector<shared_ptr<Node>> getInputs(){return vector<shared_ptr<Node>>();}
    virtual size_t getInputSize(){return 0;}
    virtual double getOldWeightForInput(string inputNodeUUID){return 0;}
    virtual void useMomentum(double momentum){}
    virtual void adjustWeights(){}
    virtual void calculateOutput()
    {
        #ifdef _DEBUG
        //cout << "Calculate output called in Node class. Doing nothing.\n";
        #endif
    }
    virtual void setOutputs(vector<shared_ptr<Node>> outputs)
    {
        /*#ifdef _DEBUG
        cout << "Inside setOutputs (base class) there are  " << outputs.size() << " outputs to be disregarded" << endl;
        #endif*/
    }
    virtual void addOutput(shared_ptr<Node> output){}
    virtual double getMomentum(){return 0;}
    virtual vector<shared_ptr<Node>> getOutputs(){return vector<shared_ptr<Node>>();}
};

#endif