#include "Node.h"

Node::Node()
{
    output = 0;
}

Node::Node(double initialOutput)
{
    output = initialOutput;
}

Node::~Node()
{}

double Node::getOutput()
{
    return output;
}

void Node::setOutput(double newOutput)
{
    output = newOutput;
}