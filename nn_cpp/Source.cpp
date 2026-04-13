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

static vector<pair<Matrix, int>> LoadData(string filename)
{
	vector <pair<Matrix, int>> data;

	ifstream file(filename);
	string line;

	while (getline(file, line))
	{
		if (line.empty()) continue;

		istringstream iss(line);
		vector<double> values;
		double val;

		while (iss >> val)
		{
			values.push_back(val);
		}

		if (values.size() == 22)
		{
			Matrix input(21, 1);
			for (int i = 0; i < 21; i++)
			{
				input[i][0] = values[i];
			}

			data.push_back({ input, (int)values[21] });
		}
	}

	file.close();
	return data;
}

static vector<pair<Matrix, int>> LoadDataNormalized(string filename, int numFeatures, int maxEntries = 1e7)
{
	vector<pair<Matrix, int>> data;
	ifstream file(filename);
	string line;

	// First pass: load raw data
	vector<vector<double>> rawValues;
	int entryCount = 0;
	while (getline(file, line))
	{
		if (line.empty()) continue;
		istringstream iss(line);
		vector<double> values;
		string token;
		while (getline(iss, token, ','))
		{
			values.push_back(stod(token));
		}
		if (values.size() == numFeatures) rawValues.push_back(values);

		if (++entryCount >= maxEntries) break;
	}
	file.close();

	// Second pass: compute min/max for normalization
	vector<double> minVals(numFeatures - 1, 1e9), maxVals(numFeatures - 1, -1e9);
	for (auto& vals : rawValues)
	{
		for (int i = 0; i < numFeatures - 1; i++)
		{
			minVals[i] = min(minVals[i], vals[i]);
			maxVals[i] = max(maxVals[i], vals[i]);
		}
	}

	// Third pass: normalize and create matrices
	for (auto& vals : rawValues)
	{
		Matrix input(numFeatures - 1, 1);
		for (int i = 0; i < numFeatures - 1; i++)
		{
			// Normalize to [0, 1]
			double normalized = (vals[i] - minVals[i]) / (maxVals[i] - minVals[i] + 1e-8);
			input[i][0] = normalized;
		}
		data.push_back({ input, (int)vals[numFeatures - 1] });
	}

	return data;
}

static void TestModel(vector<DenseLayer>& layers, const vector<pair<Matrix, int>>& testData, int classes)
{
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
		predicted += 1;  // Convert to 1-indexed

		int actual = sample.second;

		confusionMatrix[actual][predicted] += 1;

		if (predicted == actual)
		{
			correct++;
		}

		// Track TP/FP/FN for each class
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

		cout << "Class " << c << ":\n";
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

	const int numClasses = 10;
	const int numFeatures = 11;
	const int epochs = 100;
	const int batchSize = 32;
	const double learningRate = 0.01;
	const int patience = 4;
	const int maxEntries = 5000;

	// Set number of threads for matrix operations (tuning this can improve performance)
	Matrix::SetNumThreads(12);

	vector<pair<Matrix, int>> data = LoadDataNormalized("poker+hand/poker-hand-training-true.data", numFeatures, maxEntries);
	cout << "Loaded " << data.size() << " samples\n";

	// Class Distribution
	map<int, int> classCounts;
	for (auto& sample : data)
	{
		classCounts[sample.second]++;
	}

	cout << "Class Distribution\n";
	for (auto& entry : classCounts)
	{
		cout << "Class " << entry.first << ": " << entry.second << " samples (" << (100.0 * entry.second) / data.size() << "%\n";
	}

	// Calculate class weights for imbalanced data
	map<int, double> classWeights;
	double totalSamples = data.size();
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
	int numBatches = (data.size() + batchSize - 1) / batchSize;

	vector<DenseLayer> layers;
	layers.push_back(DenseLayer(numFeatures - 1, 64, Activation("RELU")));
	layers.push_back(DenseLayer(64, 64, Activation("RELU")));
	layers.push_back(DenseLayer(64, 10, Activation("SOFTMAX")));

	Loss loss((string)"CATEGORICAL_CROSS_ENTROPY", classWeights);
	int epochsWithoutImprovement = 0;
	double previousLoss = 1e9;

	for (int epoch = 0; epoch < epochs; epoch++)
	{
		std::shuffle(data.begin(), data.end(), std::mt19937(std::random_device()()));

		double totalLoss = 0;

		for (int batch = 0; batch < numBatches; batch++)
		{
			int startIdx = batch * batchSize;
			int endIdx = min(startIdx + batchSize, (int)data.size());

			for (auto& layer : layers)
			{
				layer.BeginBatch();
			}

			double batchLoss = 0;

			for (int sampleIdx = startIdx; sampleIdx < endIdx; sampleIdx++)
			{
				const auto& sample = data[sampleIdx];

				Matrix layerInput = sample.first;
				for (auto& layer : layers)
				{
					layerInput = layer.Forward(layerInput);
				}

				// One-hot encode for n classes
				Matrix target(numClasses, 1);
				for (int row = 0; row < target.Dimensions()[0]; row++) target[row][0] = 0;

				int classLabel = sample.second;
				if (sample.second == 1) target[0][0] = 1.0;
				else if (sample.second == 2) target[1][0] = 1.0;
				else target[2][0] = 1.0;

				batchLoss += loss.Forward(layerInput, target, classLabel);

				// Backpropagation
				Matrix grad = loss.Backward(layerInput, target, classLabel);
				for (int j = layers.size() - 1; j >= 0; j--)
				{
					grad = layers[j].Backward(grad, learningRate);
				}
			}

			for (auto& layer : layers)
			{
				layer.ApplyGradients(learningRate);
			}

			totalLoss += batchLoss;
		}

		if ((epoch + 1) % 1 == 0)
		{
			cout << "Epoch " << epoch + 1 << " - Loss: " << totalLoss / data.size() << "		\r";
		}

		if ((totalLoss / data.size()) < previousLoss - 0.001 > 0)
		{
			epochsWithoutImprovement = 0;
		}
		else
		{
			epochsWithoutImprovement++;
			if (epochsWithoutImprovement >= patience)
			{
				cout << "\nEarly stopping at epoch " << epoch + 1 << " with loss " << totalLoss / data.size() << endl;
				break;
			}
		}
		previousLoss = totalLoss / data.size();
	}
	cout << endl;
	cout << "Training Complete!" << endl;

	//Testing the model
	auto testData = LoadDataNormalized("poker+hand/poker-hand-testing.data", numFeatures, maxEntries);
	cout << "Loaded " << testData.size() << " Test Samples\n";

	TestModel(layers, testData, numClasses);
	
	return 0;
}