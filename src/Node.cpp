#include "Node.h"

Node::Node()
{
    ifstream uuidGetter(KERNELUUIDGEN);
    uuidGetter.open;
    if(uuidGetter.is_open())
    {
        getline(uuidGetter, uuid);
        output = 0;
        uuidGetter.close();
    }
    else
    {
        ThrowError("Cannot access kernel's UUID generator", KERNELUUIDGEN);
    }
}

Node::Node(double initialOutput): Node()
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

string Node::getUUID()
{
    return uuid;
}