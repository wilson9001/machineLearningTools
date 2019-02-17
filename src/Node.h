#ifndef NODE_H
#define NODE_H

class Node
{
    protected:
    double output;

    public:
    Node();
    Node(double initialOutput);
    ~Node();
    double getOutput();
    void setOutput(double newOutput);
};

#endif