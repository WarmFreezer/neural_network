#include <iostream>
#include <cassert>
#include <fstream>
#include <sstream>
#include <random>
#include <vector>

#include "nn.cpp"
#include "Matrix.h"

#define cout std::cout

using namespace std;
using namespace matrix;

static vector<vector<double>> LoadDataCSV(const string& filename, int numFeatures, int maxEntries = (int)1e7, bool hasHeaders = 0, char delim = ',')
{
	ifstream file(filename);
	string line;
	vector<vector<double>> result;

	if (hasHeaders)
	{
		getline(file, line);  // Skip header line
	}

	// Build category maps for non-numerical columns
	map<int, map<string, int>> categoryMaps;

	while (getline(file, line))
	{
		if (line.empty()) continue;
		istringstream iss(line);
		vector<string> tokens;
		string token;
		while (getline(iss, token, delim))
		{
			if (!token.empty() && token != "\r")
				tokens.push_back(token);
		}

		if ((int)tokens.size() != numFeatures) continue;

		vector<double> row;
		for (int col = 0; col < numFeatures; col++)
		{
			try {
				row.push_back(stod(tokens[col]));
			}
			catch (...) {
				// Nominal: assign an integer index on first sight
				auto& catMap = categoryMaps[col];
				if (catMap.find(tokens[col]) == catMap.end())
					catMap[tokens[col]] = (int)catMap.size();
				row.push_back(catMap[tokens[col]]);
			}
		}
		result.push_back(row);
		if ((int)result.size() >= maxEntries) break;
	}
	return result;
}

static pair<vector<pair<Matrix, int>>, vector<pair<Matrix, int>>> NormalizeData(
	const vector<vector<double>>& rawTrainValues,
	const vector<vector<double>>& rawTestValues,
	int numFeatures)
{
	// Compute min/max from TRAIN only � never touch test stats
	vector<double> minVals(numFeatures - 1, 1e9), maxVals(numFeatures - 1, -1e9);
	for (auto& vals : rawTrainValues)
	{
		for (int i = 0; i < numFeatures - 1; i++)
		{
			minVals[i] = min(minVals[i], vals[i]);
			maxVals[i] = max(maxVals[i], vals[i]);
		}
	}

	auto normalize = [&](const vector<vector<double>>& raw) {
		vector<pair<Matrix, int>> data;
		for (auto& vals : raw)
		{
			Matrix input(numFeatures - 1, 1);
			for (int i = 0; i < numFeatures - 1; i++)
			{
				input[i][0] = (vals[i] - minVals[i]) / (maxVals[i] - minVals[i] + 1e-8);
			}
			data.push_back({ input, (int)vals[numFeatures - 1] });
		}
		return data;
		};

	return { normalize(rawTrainValues), normalize(rawTestValues) };
}

static vector<pair<Matrix, int>> StratifiedSample(const vector<pair<Matrix, int>>& data, int samplesPerClass, int numClasses, bool zeroBased = 0)
{
	map<int, vector<pair<Matrix, int>>> byClass;
	for (auto& sample : data)
		byClass[sample.second].push_back(sample);

	vector<pair<Matrix, int>> result;

	int startClass = !zeroBased;
	int endClass = startClass + numClasses;

	for (int c = startClass; c < endClass; c++)
	{
		if (byClass.find(c) == byClass.end()) {
			cout << "Warning: class " << c << " has no samples!\n";
			continue;
		}
		auto& classData = byClass[c];
		shuffle(classData.begin(), classData.end(), mt19937(random_device()()));
		int take = min(samplesPerClass, (int)classData.size());
		if (take < samplesPerClass)
			cout << "Warning: class " << c << " only has " << take << " samples\n";

		for (int i = 0; i < take; i++)
			result.push_back(classData[i]);
	}
	// Shuffle so classes aren't in blocks
	shuffle(result.begin(), result.end(), mt19937(random_device()()));
	return result;
}

static void TestModel(vector<DenseLayer>& layers, const vector<pair<Matrix, int>>& testData, int classes, int zeroBased = 0)
{
	int startClass = !zeroBased;
	int endClass = startClass + classes;

	int correct = 0;
	map<int, int> classTP, classFP, classFN;
	Matrix confusionMatrix(classes, classes);
	for (int row = 0; row < classes; row++)
	{
		for (int col = 0; col < classes; col++)
		{
			confusionMatrix[row][col] = 0;
		}
	}

	for (auto& sample : testData)
	{
		Matrix layerInput = sample.first;
		for (auto& layer : layers)
		{
			layerInput = layer.Forward(layerInput);
		}

		// Find predicted class (argmax)
		int predicted = 0;
		double maxVal = layerInput[0][0];
		for (int i = 0; i < classes; i++)
		{
			if (layerInput[i][0] > maxVal)
			{
				maxVal = layerInput[i][0];
				predicted = i;
			}
		}
		int actual = sample.second;

		if (!zeroBased)
		{
			// predicted is already 0-based (argmax index), only actual needs converting
			actual--;
		}

		confusionMatrix[actual][predicted] += 1;

		if (predicted == actual)
		{
			correct++;
		}

		// Track TP/FP/FN for each class (both actual and predicted are now 0-based)
		for (int c = 0; c < classes; c++)
		{
			if (actual == c && predicted == c) classTP[c]++;
			else if (actual != c && predicted == c) classFP[c]++;
			else if (actual == c && predicted != c) classFN[c]++;
		}
	}

	cout << "\n=== Test Results ===\n";
	double accuracy = (double)correct / testData.size();
	cout << "Overall Accuracy: " << accuracy * 100 << "%\n\n";

	for (int c = 0; c < classes; c++)
	{
		int tp = classTP[c];
		int fp = classFP[c];
		int fn = classFN[c];

		double precision = (tp + fp > 0) ? (double)tp / (tp + fp) : 0;
		double recall = (tp + fn > 0) ? (double)tp / (tp + fn) : 0;
		double f1 = (precision + recall > 0) ? 2 * (precision * recall) / (precision + recall) : 0;

		cout << "Class " << (c + startClass) << ":\n";
		cout << "  Precision: " << precision * 100 << "%\n";
		cout << "  Recall: " << recall * 100 << "%\n";
		cout << "  F1 Score: " << f1 * 100 << "%\n\n";
	}

	Matrix::PrintMatrix(confusionMatrix);
	cout << endl;

	cout << "Testing Complete!";
}

int main()
{
	const int numClasses = 3;
	const int numFeatures = 22;
	const int epochs = 200;
	const int batchSize = 32;
	const int patience = 5;
	const int maxEntries = 5000;
	const bool zeroBasedClasses = false;
	const bool hasHeaders = false;
	const char delim = ' ';
	const string trainFile = "thyroid+disease/ann-train.data";
	const string testFile = "thyroid+disease/ann-test.data";

	// Set number of threads for matrix operations (tuning this can improve performance)
	Matrix::SetNumThreads(12);

	vector<vector<double>> data = LoadDataCSV(trainFile, numFeatures, maxEntries, hasHeaders, delim);
	vector<vector<double>> testData = LoadDataCSV(testFile, numFeatures, maxEntries, hasHeaders, delim);
	
	auto [trainDataNorm, testDataNorm] = NormalizeData(data, testData, numFeatures);
	
	auto trainSample = StratifiedSample(trainDataNorm, 500, numClasses, zeroBasedClasses);
	auto testSample = StratifiedSample(testDataNorm, 500, numClasses, zeroBasedClasses);

	cout << "Loaded " << trainSample.size() << " samples\n";

	// Class Distribution
	map<int, int> classCounts;
	for (auto& sample : trainSample)
	{
		classCounts[sample.second]++;
	}

	// Calculate class weights for imbalanced data
	map<int, double> classWeights;
	double totalSamples = trainSample.size();
	for (auto& entry : classCounts)
	{
		classWeights[entry.first] = totalSamples / (numClasses * entry.second);
	}
	cout << "Class Weights\n";
	for (auto& w : classWeights)
	{
		cout << "Class " << w.first << ": " << w.second << "\n";
	}

	// Training the model
	int numBatches = (trainSample.size() + batchSize - 1) / batchSize;

	vector<DenseLayer> layers;
	layers.push_back(DenseLayer(numFeatures - 1, 64, Activation("RELU")));
	layers.push_back(DenseLayer(64, 64, Activation("RELU")));
	layers.push_back(DenseLayer(64, numClasses, Activation("SOFTMAX")));

	Loss loss((string)"CATEGORICAL_CROSS_ENTROPY", classWeights);
	int epochsWithoutImprovement = 0;
	double previousLoss = 1e9;

	for (int epoch = 0; epoch < epochs; epoch++)
	{
		std::shuffle(trainSample.begin(), trainSample.end(), std::mt19937(std::random_device()()));

		double totalLoss = 0;

		for (int batch = 0; batch < numBatches; batch++)
		{
			int startIdx = batch * batchSize;
			int endIdx = min(startIdx + batchSize, (int)trainSample.size());

			for (auto& layer : layers)
			{
				layer.BeginBatch();
			}

			double batchLoss = 0;

			for (int sampleIdx = startIdx; sampleIdx < endIdx; sampleIdx++)
			{
				const auto& sample = trainSample[sampleIdx];

				Matrix layerInput = sample.first;
				for (auto& layer : layers)
				{
					layerInput = layer.Forward(layerInput);
				}

				int classLabel = sample.second;

				Matrix target(numClasses, 1);
				for (int row = 0; row < numClasses; row++) 
				{
					target[row][0] = 0.0;
				}

				if (!zeroBasedClasses)
				{
					classLabel--;
				}

				target[classLabel][0] = 1.0;  

				batchLoss += loss.Forward(layerInput, target, classLabel);

				// Backpropagation
				Matrix grad = loss.Backward(layerInput, target, classLabel);
				for (int j = layers.size() - 1; j >= 0; j--)
				{
					grad = layers[j].Backward(grad);
				}
			}

			for (auto& layer : layers)
			{
				layer.ApplyGradients();
			}

			totalLoss += batchLoss;
		}

		if ((epoch + 1) % 1 == 0)
		{
			cout << "Epoch " << epoch + 1 << " - Loss: " << totalLoss / trainSample.size() << "		\r";
		}

		if ((totalLoss / trainSample.size()) < previousLoss - 0.001)
		{
			epochsWithoutImprovement = 0;
		}
		else
		{
			epochsWithoutImprovement++;
			if (epochsWithoutImprovement >= patience)
			{
				cout << "\nEarly stopping at epoch " << epoch + 1 << " with loss " << totalLoss / trainSample.size() << endl;
				break;
			}
		}
		previousLoss = totalLoss / trainSample.size();
	}
	cout << endl;
	cout << "Training Complete!" << endl;

	//Testing the model
	cout << "Loaded " << testSample.size() << " Test Samples\n";

	TestModel(layers, testSample, numClasses);
	
	return 0;
}