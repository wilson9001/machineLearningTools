#ifndef NODE_H
#define NODE_H

#include <string>
#include <fstream>
#include "error.h"

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
};

#endif