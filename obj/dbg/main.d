../obj/dbg/main.o: main.cpp learner.h matrix.h rand.h baseline.h error.h filter.h \
 perceptron.h NeuralNet.h Layer.h MiddleNode.h NonInputNode.h Node.h \
 DecisionTree.h DTNode.h KNN.h
	g++ -Wall -g -D_DEBUG -c main.cpp -o ../obj/dbg/main.o
