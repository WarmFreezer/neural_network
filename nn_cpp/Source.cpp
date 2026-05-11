#include <iostream>
#include <cassert>
#include <fstream>
#include <sstream>
#include <random>
#include <vector>
#include <chrono>
#include <array>

#include "nn.cpp"
#include "Matrix.h"

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
	static mt19937 rng(random_device{}());
	map<int, vector<pair<Matrix, int>>> byClass;
	for (auto& sample : data)
		byClass[sample.second].push_back(sample);

	vector<pair<Matrix, int>> result;

	int startClass = !zeroBased;
	int endClass = startClass + numClasses;

	for (int c = startClass; c < endClass; c++)
	{
		if (byClass.find(c) == byClass.end()) 
		{
			std::cout << "Warning: class " << c << " has no samples!\n";
			continue;
		}
		auto& classData = byClass[c];
		shuffle(classData.begin(), classData.end(), rng);
		int take = min(samplesPerClass, (int)classData.size());
		if (take < samplesPerClass)
			std::cout << "Warning: class " << c << " only has " << take << " samples\n";

		for (int i = 0; i < take; i++)
			result.push_back(classData[i]);
	}
	// Shuffle so classes aren't in blocks
	shuffle(result.begin(), result.end(), rng);
	return result;
}

static void TestModel(DenseLayer* layers, int numLayers, const vector<pair<Matrix, int>>& testData, int classes, int zeroBased = 0)
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
		for (int layer = 0; layer < numLayers; layer++)
		{
			layerInput = layers[layer].Forward(layerInput);
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

	std::cout << "\n=== Test Results ===\n";
	double accuracy = (double)correct / testData.size();
	std::cout << "Overall Accuracy: " << accuracy * 100 << "%\n\n";

	for (int c = 0; c < classes; c++)
	{
		int tp = classTP[c];
		int fp = classFP[c];
		int fn = classFN[c];

		double precision = (tp + fp > 0) ? (double)tp / (tp + fp) : 0;
		double recall = (tp + fn > 0) ? (double)tp / (tp + fn) : 0;
		double f1 = (precision + recall > 0) ? 2 * (precision * recall) / (precision + recall) : 0;

		std::cout << "Class " << (c + startClass) << ":\n";
		std::cout << "  Precision: " << precision * 100 << "%\n";
		std::cout << "  Recall: " << recall * 100 << "%\n";
		std::cout << "  F1 Score: " << f1 * 100 << "%\n\n";
	}

	Matrix::PrintMatrix(confusionMatrix);
	std::cout << endl;

	std::cout << "Testing Complete!";
}

int main()
{
	const int numClasses = 3;
	const int numFeatures = 22;
	const int epochs = 200;
	const int batchSize = 128;
	const int patience = 20;
	const double negligibleImprovement = 0.0001;
	const int maxEntries = 5000;
	const bool zeroBasedClasses = false;
	const bool hasHeaders = false;
	const char delim = ' ';
	const string trainFile = "thyroid+disease/ann-train.data";
	const string testFile = "thyroid+disease/ann-test.data";
	const int numThreads = 10;

	omp_set_num_threads(numThreads);

	vector<vector<double>> data = LoadDataCSV(trainFile, numFeatures, maxEntries, hasHeaders, delim);
	vector<vector<double>> testData = LoadDataCSV(testFile, numFeatures, maxEntries, hasHeaders, delim);
	
	auto [trainDataNorm, testDataNorm] = NormalizeData(data, testData, numFeatures);
	
	auto trainSample = StratifiedSample(trainDataNorm, 1500, numClasses, zeroBasedClasses);
	auto testSample = StratifiedSample(testDataNorm, 1500, numClasses, zeroBasedClasses);

	std::cout << "Loaded " << trainSample.size() << " samples\n";

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
	std::cout << "\nClass Weights\n";
	for (auto& w : classWeights)
	{
		std::cout << "Class " << w.first << ": " << w.second << "\n";
	}
	std::cout << endl;

	// Training the model
	int numBatches = (trainSample.size() + batchSize - 1) / batchSize;

	DenseLayer layers[] = {
		DenseLayer(numFeatures - 1, 64, Activation("RELU")),
		DenseLayer(64, 64, Activation("RELU")),
		DenseLayer(64, numClasses, Activation("SOFTMAX"))
	};

	Loss loss((string)"CATEGORICAL_CROSS_ENTROPY", classWeights);
	int epochsWithoutImprovement = 0;
	double previousLoss = 1e9;

	static mt19937 rng(random_device{}());

	epochsWithoutImprovement = 0;
	previousLoss = 1e9;

	const int numLayers = (int)size(layers);
	std::vector<std::vector<DenseLayer>> threadLayers(numThreads);
	for (int t = 0; t < numThreads; t++) {
		for (const auto& layer : layers) {
			threadLayers[t].push_back(layer);
		}
	}

	// Start Time
	auto start = std::chrono::high_resolution_clock::now();
	for (int epoch = 0; epoch < epochs; epoch++)
	{
		shuffle(trainSample.begin(), trainSample.end(), rng);

		double totalLoss = 0;

		for (int batch = 0; batch < numBatches; batch++)
		{
			int startIdx = batch * batchSize;
			int endIdx = min(startIdx + batchSize, (int)trainSample.size());

			// Sync weights from updated main layers and reset gradient accumulators.
			for (auto& tl : threadLayers)
				for (int j = 0; j < numLayers; j++) { tl[j].SyncWeightsFrom(layers[j]); tl[j].BeginBatch(); }

			double batchLoss = 0;

#pragma omp parallel for reduction(+:batchLoss) schedule(static)
			for (int sampleIdx = startIdx; sampleIdx < endIdx; sampleIdx++)
			{
				int tid = omp_get_thread_num();
				auto& localLayers = threadLayers[tid];
				const auto& sample = trainSample[sampleIdx];

				// Forward pass on thread-local layer copies
				Matrix currentInput = sample.first;
				for (int li = 0; li < numLayers; li++)
					currentInput = localLayers[li].Forward(currentInput);

				int originalLabel = sample.second;
				int classIdx = zeroBasedClasses ? originalLabel : originalLabel - 1;

				Matrix target(numClasses, 1);
				target[classIdx][0] = 1.0;

				batchLoss += loss.Forward(currentInput, target, originalLabel);

				// Backward pass on thread-local layer copies
				Matrix grad = loss.Backward(currentInput, target, originalLabel);
				for (int j = numLayers - 1; j >= 0; j--)
					grad = localLayers[j].Backward(grad);
			}

			// Merge per-thread gradient accumulators into main layers, then update
			for (auto& layer : layers) layer.BeginBatch();
			for (auto& tl : threadLayers)
				for (int j = 0; j < numLayers; j++)
					layers[j].AccumulateFrom(tl[j]);

			for (auto& layer : layers) layer.ApplyGradients();

			totalLoss += batchLoss;
		}

		double epochLoss = totalLoss / trainSample.size();
		std::cout << "Epoch " << epoch + 1 << " - Loss: " << epochLoss << "		\r";

		if (epochLoss < previousLoss - negligibleImprovement)
		{
			epochsWithoutImprovement = 0;
		}
		else
		{
			epochsWithoutImprovement++;
			if (epochsWithoutImprovement >= patience)
			{
				std::cout << "Early stopping at epoch " << epoch + 1 << " with loss " << epochLoss << endl;
				break;
			}
		}
		previousLoss = epochLoss;
	}
	auto end = std::chrono::high_resolution_clock::now();
	// End Time

	std::cout << "\nTraining completed in: " << (end - start).count() / 1000000000.00 << " Seconds\n";

	//Testing the model
	std::cout << "Loaded " << testSample.size() << " Test Samples\n";

	TestModel(layers, size(layers), testSample, numClasses);
	
	return 0;
}