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

static vector<pair<Matrix, int>> LoadDataNormalized(string filename)
{
	vector<pair<Matrix, int>> data;
	ifstream file(filename);
	string line;

	// First pass: load raw data
	vector<vector<double>> rawValues;
	while (getline(file, line))
	{
		if (line.empty()) continue;
		istringstream iss(line);
		vector<double> values;
		double val;
		while (iss >> val) values.push_back(val);
		if (values.size() == 22) rawValues.push_back(values);
	}
	file.close();

	// Second pass: compute min/max for normalization
	vector<double> minVals(21, 1e9), maxVals(21, -1e9);
	for (auto& vals : rawValues)
	{
		for (int i = 0; i < 21; i++)
		{
			minVals[i] = min(minVals[i], vals[i]);
			maxVals[i] = max(maxVals[i], vals[i]);
		}
	}

	// Third pass: normalize and create matrices
	for (auto& vals : rawValues)
	{
		Matrix input(21, 1);
		for (int i = 0; i < 21; i++)
		{
			// Normalize to [0, 1]
			double normalized = (vals[i] - minVals[i]) / (maxVals[i] - minVals[i] + 1e-8);
			input[i][0] = normalized;
		}
		data.push_back({ input, (int)vals[21] });
	}

	return data;
}

static void TestModel(vector<DenseLayer>& layers, const vector<pair<Matrix, int>>& testData)
{
	int correct = 0;
	map<int, int> classTP, classFP, classFN;

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
		for (int i = 1; i < 3; i++)
		{
			if (layerInput[i][0] > maxVal)
			{
				maxVal = layerInput[i][0];
				predicted = i;
			}
		}
		predicted += 1;  // Convert to 1-indexed

		int actual = sample.second;

		if (predicted == actual)
		{
			correct++;
		}

		// Track TP/FP/FN for each class
		for (int c = 1; c <= 3; c++)
		{
			if (actual == c && predicted == c) classTP[c]++;
			else if (actual != c && predicted == c) classFP[c]++;
			else if (actual == c && predicted != c) classFN[c]++;
		}
	}

	cout << "\n=== Test Results ===\n";
	double accuracy = (double)correct / testData.size();
	cout << "Overall Accuracy: " << accuracy * 100 << "%\n\n";

	for (int c = 1; c <= 3; c++)
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

	cout << "Testing Complete!";
}

int main()
{
	int epochs = 100;
	int batchSize = 32;
	double learningRate = 0.01;

	// Set number of threads for matrix operations (tuning this can improve performance)
	Matrix::SetNumThreads(12);

	vector<pair<Matrix, int>> data = LoadDataNormalized("Thyroid/ann-train.data");
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
	int numClasses = classCounts.size();
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
	layers.push_back(DenseLayer(21, 64, Activation("RELU")));
	layers.push_back(DenseLayer(64, 64, Activation("RELU")));
	layers.push_back(DenseLayer(64, 3, Activation("SOFTMAX")));
	 
	Loss loss((string)"CATEGORICAL_CROSS_ENTROPY", classWeights);

	for (int epoch = 0; epoch < epochs; epoch++)
	{
		std::shuffle(data.begin(), data.end(), std::mt19937(std::random_device()()));

		double totalLoss = 0;
		int batchCount = 0;

		for (int batch = 0; batch < numBatches; batch++)
		{
			vector<Matrix> batchGradients(0);
			double batchLoss = 0;

			int startIdx = batch * batchSize;
			int endIdx = min(startIdx + batchSize, (int)data.size());

			for (int sampleIdx = startIdx; sampleIdx < endIdx; sampleIdx++)
			{
				auto sample = data[sampleIdx];

				Matrix layerInput = sample.first;
				for (auto& layer : layers)
				{
					layerInput = layer.Forward(layerInput);
				}

				// One-hot encode for 3 classes
				Matrix target(3, 1);
				target[0][0] = 0.0;
				target[1][0] = 0.0;
				target[2][0] = 0.0;

				int classLabel = sample.second;
				if (sample.second == 1)
					target[0][0] = 1.0;
				else if (sample.second == 2)
					target[1][0] = 1.0;
				else
					target[2][0] = 1.0;

				batchLoss += loss.Forward(layerInput, target, classLabel);

				// Backpropagation
				Matrix grad = loss.Backward(layerInput, target, classLabel);
				batchGradients.push_back(grad);
			}

			Matrix avgGrad = batchGradients[0];
			for (int i = 1; i < batchGradients.size(); i++)
			{
				for (int row = 0; row < avgGrad.Dimensions()[0]; row++)
				{
					avgGrad[row][0] += batchGradients[i][row][0];
				}
			}
			for (int row = 0; row < avgGrad.Dimensions()[0]; row++)
			{
				avgGrad[row][0] /= batchGradients.size();
			}

			// Backprop with averaged gradient
			Matrix grad = avgGrad;
			for (int j = layers.size() - 1; j >= 0; j--)
			{
				grad = layers[j].Backward(grad, learningRate);
			}

			totalLoss += batchLoss;
			batchCount++;
		}

		if ((epoch + 1) % 1 == 0)
		{
			cout << "Epoch " << epoch + 1 << " - Loss: " << totalLoss / data.size() << endl;
		}
	}

	cout << "Training Complete!" << endl;

	//Testing the model
	auto testData = LoadDataNormalized("Thyroid/ann-test.data");
	cout << "Loaded " << testData.size() << " Test Samples\n";

	TestModel(layers, testData);

	return 0;
}