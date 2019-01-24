// ----------------------------------------------------------------
// The contents of this file are distributed under the CC0 license.
// See http://creativecommons.org/publicdomain/zero/1.0/
// ----------------------------------------------------------------

#include "learner.h"
#include "baseline.h"
#include "error.h"
#include "rand.h"
#include "filter.h"
#include <iostream>
#include <fstream>
#include <map>
#include <vector>
#include <algorithm>
#include <sstream>
#include <memory>
#include <cstring>
#include <time.h>
#include <math.h>
#ifdef WIN32
# include <windows.h>
# include <float.h>
#else // WIN32
# include <sys/time.h>
# include <fenv.h>
#endif // else WIN32


using namespace std;

class ArgParser
{
	string arff;
	string learner;
	string evaluation;
	char* evalExtra;
	bool normalize;
	bool nominal_to_cat;
	bool discretize;
	unsigned int seed;

public:
	//You may need to add more options for specific learning models
	ArgParser ( char* argv[], int argc )
	{
		arff = "";
		learner = "";
		evaluation = "";
		seed = (unsigned int)time ( NULL );
		normalize = false;
		nominal_to_cat = false;
		discretize = false;
		for ( int i = 1; i < argc; i++ )
		{
			if ( strcmp ( argv[i], "-A" ) == 0 )
			{
				arff = argv[++i];
			}
			else if ( strcmp ( argv[i], "-L" ) == 0 )
			{
				learner = argv[++i];
			}
			else if ( strcmp ( argv[i], "-E" ) == 0 )
			{
				evaluation = argv[++i];
				if ( strcmp ( argv[i], "static" ) == 0 )
					evalExtra = argv[++i]; //expecting a test set name
				else if ( strcmp ( argv[i], "random" ) == 0 )
					evalExtra = argv[++i]; //expecting a double representing the percentage for training. Note stratification is NOT done
				else if ( strcmp ( argv[i], "cross" ) == 0 )
					
					evalExtra = argv[++i]; //expecting the number of folds
				else if ( strcmp ( argv[i], "training" ) != 0 )
					ThrowError ( "Invalid Evaluation Method: ", argv[i] );
			}
			else if ( strcmp ( argv[i], "-N" ) == 0 )
				normalize = true;
			else if ( strcmp ( argv[i], "-C" ) == 0 )
				nominal_to_cat = true;
			else if ( strcmp ( argv[i], "-D" ) == 0 )
				discretize = true;
			else if ( strcmp ( argv[i], "-R" ) == 0 )
				seed = atoi ( argv[++i] );
			else
				ThrowError ( "Invalid paramater: ", argv[i] );
		}
		if ( arff == "" || learner == "" || evaluation == "" )
		{
			cout << "Missing parameters.  Usage:\n"
			<< "MLSystemManager -L [learningAlgorithm] -A [ARFF_File] -E [EvaluationMethod] {[ExtraParamters]} [-N] [-R seed]\n\n"
			<< "Possible evaluation methods are:\n"
			<< "MLSystemManager -L [learningAlgorithm] -A [ARFF_File] -E training\n"
			<< "MLSystemManager -L [learningAlgorithm] -A [ARFF_File] -E static [TestARFF_File]\n"
			<< "MLSystemManager -L [learningAlgorithm] -A [ARFF_File] -E random [PercentageForTraining]\n"
			<< "MLSystemManager -L [learningAlgorithm] -A [ARFF_File] -E cross [numOfFolds]\n\n";
			ThrowError ( "Missing parameters" );
		}
	}

	//The getter methods
	string getARFF() { return arff; }
	string getLearner() { return learner; }
	string getEvaluation() { return evaluation; }
	char* getEvalExtra() { return evalExtra; }
	bool getNormal() { return normalize; }
	bool getNominalToCat() { return nominal_to_cat; }
	bool getDiscretize() { return discretize; }
	unsigned int getSeed() { return seed; }
};

// Returns the number of seconds since some fixed point in the past, with at least millisecond precision
double getTime()
{
#ifdef WIN32
	time_t t;
	SYSTEMTIME st;
	GetSystemTime(&st);
	return ((double)st.wMilliseconds * 1e-3 + time(&t));
#else
	struct timeval tp;
	gettimeofday(&tp, NULL);
	return ((double)tp.tv_sec + (double)tp.tv_usec * 1e-6);
#endif
}

SupervisedLearner* getLearner(string model, Rand& r)
{
	if (model.compare("baseline") == 0)
		return new BaselineLearner(r);
	else if (model.compare("neuralnet") == 0)
		ThrowError("Sorry, ", model, " is not yet implemented");
	else if (model.compare("decisiontree") == 0)
		ThrowError("Sorry, ", model, " is not yet implemented");
	else if (model.compare("naivebayes") == 0)
		ThrowError("Sorry, ", model, " is not yet implemented");
	else if (model.compare("knn") == 0)
		ThrowError("Sorry, ", model, " is not yet implemented");
	else
		ThrowError("Unrecognized model: ", model);
	return NULL;
}

void doit(ArgParser& parser)
{
	// Load the model
	Rand r ( parser.getSeed() );
	string model = parser.getLearner();
	SupervisedLearner* learner = getLearner ( model, r );

	// Wrap the learner with the specified filters
	if ( parser.getNominalToCat() )
		learner = new NominalToCategorical( learner );
	if ( parser.getNormal() )
		learner = new Normalize( learner );
	if ( parser.getDiscretize() )
		learner = new Discretize( learner );

	// This line says to "delete(learner)" when this object (ap_learner) goes out of scope. This
	// technique is better than just calling "delete(learner)" ourselves at the end
	// of this method because this will clean up memory even if an exception is thrown.
	auto_ptr<SupervisedLearner> ap_learner ( learner );

	// Load the ARFF file
	string fileName = parser.getARFF();
	Matrix dataset;
	dataset.loadARFF ( fileName );
	size_t labelDims = 1;

	// Display some values
	cout << "Dataset name: " << fileName << endl;
	cout << "Number of instances (rows): " << dataset.rows() << endl;
	cout << "Number of attributes (cols): " << dataset.cols() << endl;
	cout << "Learning algorithm: " << model << endl;
	string evaluation = parser.getEvaluation();
	cout << "Evaluation method: " << evaluation << endl;
	if ( evaluation.compare ( "training" ) == 0 )
	{
		// Train
		Matrix trainFeatures, trainLabels;
		trainFeatures.copyPart(dataset, 0, 0, dataset.rows(), dataset.cols() - labelDims);
		trainLabels.copyPart(dataset, 0, dataset.cols() - labelDims, dataset.rows(), labelDims);
		double timeBeforeTraining = getTime();
		learner->train(trainFeatures, trainLabels);
		double timeAfterTraining = getTime();

		// Test on the same dataset
		Matrix stats;
		double timeBeforeTesting = getTime();
		double accuracy = learner->measureAccuracy(trainFeatures, trainLabels, &stats);
		double timeAfterTesting = getTime();

		// Print results
		cout << "\n\nAccuracy on the training set: (does NOT imply the ability to generalize)\n";
		for(size_t i = 0; i < stats.cols(); i++)
			cout << dataset.attrValue(dataset.cols() - 1, i) << ": " << stats[0][i] << "/" << stats[1][i] << "\n";
		cout << "Set accuracy: " << accuracy << "\n";
		cout<< "\nTraining time: " << (timeAfterTraining - timeBeforeTraining) << " seconds\n";
		cout<< "\nTesting time: " << (timeAfterTesting - timeBeforeTesting) << " seconds\n";
		cout.flush();
	}
	else if ( evaluation.compare ( "static" ) == 0 )
	{
		// Train
		Matrix trainFeatures, trainLabels;
		trainFeatures.copyPart(dataset, 0, 0, dataset.rows(), dataset.cols() - labelDims);
		trainLabels.copyPart(dataset, 0, dataset.cols() - labelDims, dataset.rows(), labelDims);
		double timeBeforeTraining = getTime();
		learner->train(trainFeatures, trainLabels);
		double timeAfterTraining = getTime();

		// Test on the same dataset
		Matrix stats;
		double accuracy = learner->measureAccuracy(trainFeatures, trainLabels, &stats);

		// Print results
		cout << "\n\nAccuracy on the training set: (does NOT imply the ability to generalize)\n";
		for(size_t i = 0; i < stats.cols(); i++)
			cout << dataset.attrValue(dataset.cols() - 1, i) << ": " << stats[0][i] << "/" << stats[1][i] << "\n";
		cout << "Set accuracy: " << accuracy << "\n";
		cout.flush();

		// Test on the test set
		if(!parser.getEvalExtra())
			ThrowError("Expected a test dataset to be specified");
		string testSetFilename = parser.getEvalExtra();
		Matrix testSet;
		testSet.loadARFF(testSetFilename);
		dataset.checkCompatibility(testSet);
		Matrix testFeatures, testLabels;
		testFeatures.copyPart(testSet, 0, 0, testSet.rows(), testSet.cols() - labelDims);
		testLabels.copyPart(testSet, 0, testSet.cols() - labelDims, testSet.rows(), labelDims);
		double timeBeforeTesting = getTime();
		accuracy = learner->measureAccuracy(testFeatures, testLabels, &stats);
		double timeAfterTesting = getTime();

		// Print results
		cout << "\n\nAccuracy on the test set:\n";
		for(size_t i = 0; i < stats.cols(); i++)
			cout << dataset.attrValue(dataset.cols() - 1, i) << ": " << stats[0][i] << "/" << stats[1][i] << "\n";
		cout << "Set accuracy: " << accuracy << "\n";
		cout << "\nTraining time: " << (timeAfterTraining - timeBeforeTraining) << " seconds\n";
		cout << "\nTesting time: " << (timeAfterTesting - timeBeforeTesting) << " seconds\n";
		cout.flush();
	}
	else if ( evaluation.compare ( "random" ) == 0 )
	{
		// Split the data
		if(!parser.getEvalExtra())
			ThrowError("Expected the training percentage to be specified");
		double trainPercent = atof ( parser.getEvalExtra() );
		if ( trainPercent < 0.0 || trainPercent > 1.0 )
			ThrowError("Expected the percentage to be between 0 and 1\n");
		size_t trainRows = (size_t)floor(dataset.rows() * trainPercent + 0.5);
		dataset.shuffleRows(r);
		Matrix trainSet, testSet;
		trainSet.copyPart(dataset, 0, 0, trainRows, dataset.cols());
		testSet.copyPart(dataset, trainRows, 0, dataset.rows() - trainRows, dataset.cols());

		// Train
		Matrix trainFeatures, trainLabels;
		trainFeatures.copyPart(trainSet, 0, 0, trainSet.rows(), trainSet.cols() - labelDims);
		trainLabels.copyPart(trainSet, 0, trainSet.cols() - labelDims, trainSet.rows(), labelDims);
		double timeBeforeTraining = getTime();
		learner->train(trainFeatures, trainLabels);
		double timeAfterTraining = getTime();

		// Test on the same dataset
		Matrix stats;
		double accuracy = learner->measureAccuracy(trainFeatures, trainLabels, &stats);

		// Print results
		cout << "\n\nAccuracy on the training set: (does NOT imply the ability to generalize)\n";
		for(size_t i = 0; i < stats.cols(); i++)
			cout << dataset.attrValue(dataset.cols() - 1, i) << ": " << stats[0][i] << "/" << stats[1][i] << "\n";
		cout << "Set accuracy: " << accuracy << "\n";
		cout.flush();

		// Test on the test set
		Matrix testFeatures, testLabels;
		testFeatures.copyPart(testSet, 0, 0, testSet.rows(), testSet.cols() - labelDims);
		testLabels.copyPart(testSet, 0, testSet.cols() - labelDims, testSet.rows(), labelDims);
		double timeBeforeTesting = getTime();
		accuracy = learner->measureAccuracy(testFeatures, testLabels, &stats);
		double timeAfterTesting = getTime();

		// Print results
		cout << "\n\nAccuracy on the test set:\n";
		for(size_t i = 0; i < stats.cols(); i++)
			cout << dataset.attrValue(dataset.cols() - 1, i) << ": " << stats[0][i] << "/" << stats[1][i] << "\n";
		cout << "Set accuracy: " << accuracy << "\n";
		cout<< "\nTraining time: " << (timeAfterTraining - timeBeforeTraining) << " seconds\n";
		cout<< "\nTesting time: " << (timeAfterTesting - timeBeforeTesting) << " seconds\n";
		cout.flush();
	}
	else if ( evaluation.compare ( "cross" ) == 0 )
	{
		if(!parser.getEvalExtra())
			ThrowError("Expected the number of folds to be specified");
		size_t folds = atoi ( parser.getEvalExtra() );
		Matrix features, labels;
		features.copyPart(dataset, 0, 0, dataset.rows(), dataset.cols() - labelDims);
		labels.copyPart(dataset, 0, dataset.cols() - labelDims, dataset.rows(), labelDims);
		double accuracy = learner->crossValidate(1, folds, features, labels, r, true);
		if(labels.valueCount(0) == 0)
			cout << "Root Mean Squared Error: " << accuracy << "\n";
		else
			cout << "Mean predictive accuracy: " << accuracy << "\n";
		cout.flush();
	}
}

int main(int argc, char *argv[])
{
	// Swallowing errors leads to painful debugging experiences--never swallow errors.
#ifdef WIN32
	// Don't silently swallow floating point errors on Windows.
	unsigned int cw = _control87(0, 0) & MCW_EM;
	cw &= ~(_EM_INVALID | _EM_ZERODIVIDE | _EM_OVERFLOW);
	_control87(cw, MCW_EM);
#else
#	ifdef DARWIN
	// Anyone know how to tell Darwin to break if there is a floating point error?
#	else
	// Don't silently swallow floating point errors on Linux.
	feenableexcept(FE_INVALID | FE_DIVBYZERO | FE_OVERFLOW);
#	endif
#endif

	int ret = 0;
	try
	{
		// parse the args
		ArgParser parser(argv, argc);

		// do what they say to do
		doit(parser);
	}
	catch(const std::exception& e)
	{
		cerr << "Error: " << e.what() << "\n";
		ret = 1;
	}

	return ret;
}
