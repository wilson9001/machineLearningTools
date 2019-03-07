../obj/opt/main.o: main.cpp learner.h matrix.h rand.h baseline.h error.h filter.h \
 perceptron.h NeuralNet.h Layer.h MiddleNode.h NonInputNode.h Node.h \
 DecisionTree.h DTNode.h
	g++ -Wall -O3 -c main.cpp -o ../obj/opt/main.o
