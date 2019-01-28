../obj/opt/main.o: main.cpp learner.h matrix.h rand.h baseline.h error.h filter.h \
 perceptron.h
	g++ -Wall -O3 -c main.cpp -o ../obj/opt/main.o
