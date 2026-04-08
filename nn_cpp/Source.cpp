#include <iostream>
#include <cassert>
#include <fstream>
#include <sstream>

#include "nn.cpp"
#include "Matrix.h"

using namespace std;
using namespace matrix;

vector<pair<Matrix, int>> LoadData(string filename)
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

int main()
{
	vector<pair<Matrix, int>> data = LoadData("Thyroid/ann-train.data");
	cout << "Loaded " << data.size() << " samples\n";

	vector<DenseLayer> layers;
	layers.push_back(DenseLayer(21, 64, Activation("RELU")));
	layers.push_back(DenseLayer(64, 64, Activation("RELU")));
	layers.push_back(DenseLayer(64, 2, Activation("SIGMOID")));
	 
	Loss loss("MSE");
	double learningRate(0.1);
	int epochs = 100;

	for (int epoch = 0; epoch < epochs; epoch++)
	{
		double totalLoss = 0;
		for (auto& sample : data)
		{
			Matrix layerInput = sample.first;
			for (auto& layer : layers)
			{
				layerInput = layer.Forward(layerInput);
			}

			Matrix target(2, 1);
			if (sample.second == 2)
			{
				target[0][0] = 1.0;
				target[1][0] = 0.0;
			}
			else
			{
				target[0][0] = 0.0;
				target[1][0] = 1.0;
			}

			totalLoss += loss.Forward(layerInput, target);

			Matrix grad = loss.Backward(layerInput, target);
			for (int j = layers.size() - 1; j >= 0; j--)
			{
				grad = layers[j].Backward(grad, learningRate);
			}
		}

		if ((epoch + 1) % 1 == 0)
		{
			cout << "Epoch " << epoch + 1 << " - Loss: " << totalLoss / data.size() << endl;
		}
	}

	cout << "Training Complete!" << endl;

	//Testing the model
	auto testData = LoadData("Thyroid/ann-test.data");
	cout << "Loaded " << testData.size() << " Test Samples\n";

	int correct = 0; 
	int tp2 = 0, fp2 = 0, fn2 = 0;

	for (auto& sample : testData)
	{
		Matrix layerInput = sample.first;
		for (auto& layer : layers)
		{
			layerInput = layer.Forward(layerInput);
		}

		int predicted = layerInput[0][0] > layerInput[1][0] ? 2 : 3;
		int actual = sample.second;

		if (predicted == actual)
		{
			correct++;
		}

		if (actual == 2)
		{
			if (predicted == 2) tp2++;
			else fn2++;
		}
		else
		{
			if (predicted == 2) fp2++;
		}
	}

	double accuracy = (double)correct / testData.size();
	double precision2 = tp2 + fp2 > 0 ? (double)tp2 / (tp2 + fp2) : 0;
	double recall2 = tp2 + fn2 > 0 ? (double)tp2 / (tp2 + fn2) : 0;
	double f1 = precision2 + recall2 > 0 ? 2 * (precision2 * recall2) / (precision2 + recall2) : 0;
	
	cout << "Test Accuracy: " << accuracy * 100 << "%\n";
	cout << "Class 2 - Precision: " << precision2 * 100 << "%, Recall: " << recall2 * 100 << "%, F1 Score: " << f1 * 100 << "%\n";
	cout << "Class 2 - Recall: " << recall2 * 100 << "%\n";
	cout << "f1 Score: " << f1 * 100 << "%\n";

	cout << "Testing Complete!";

	return 0;
}