#pragma once
#include <iostream>
#include <vector>
#include <omp.h>
#include <map>

#include "Matrix.h"

#define Matrix matrix::Matrix

enum class ActivationType { RELU, SIGMOID, TANH, SOFTMAX, None };

class LinearLayer
{
public: 
	Matrix weights;
	Matrix bias;
	Matrix lastInput;

	LinearLayer(int in, int out)
	{
		weights = Matrix(out, in);
		bias = Matrix(out, 1);
		Matrix::PopulateRand(weights);
		Matrix::PopulateRand(bias);
	}

	Matrix Forward(Matrix& input)
	{
		lastInput = input;
		return (weights * input) + bias;
	}

	Matrix Backward(Matrix& grad, double learningRate)
	{
		Matrix weightGrad = grad * Matrix::Transpose(lastInput);
		Matrix biasGrad = grad;
		weights = weights - (learningRate * weightGrad);
		bias = bias - (learningRate * biasGrad);

		return Matrix::Transpose(weights) * grad;
	}
};

class Activation
{
public:
	ActivationType type;
	Matrix lastInput;
	Matrix lastOutput;

	Activation()
	{
		type = ActivationType::None;
	}

	Activation(std::string type)
	{
		if (type == "RELU")
		{
			this->type = ActivationType::RELU;
		}
		else if (type == "SIGMOID")
		{
			this->type = ActivationType::SIGMOID;
		}
		else if (type == "TANH")
		{
			this->type = ActivationType::TANH;
		}
		else if (type == "SOFTMAX")
		{
			this->type = ActivationType::SOFTMAX;
		}
		else
		{
			this->type = ActivationType::None;
		}
	}

	Matrix Forward(const Matrix& input)
	{
		lastInput = input;
		switch (type)
		{
		case ActivationType::RELU:
			lastOutput = ReLU(input);
			break;
		case ActivationType::SIGMOID:
			lastOutput = Sigmoid(input);
			break;
		case ActivationType::TANH:
			lastOutput = TanH(input);
			break;
		case ActivationType::SOFTMAX:
			lastOutput = Softmax(input);
			break;
		default:
			lastOutput = input;
			break;
		}
		return lastOutput;
	}

	Matrix Backward(const Matrix& grad)
	{
		switch (type)
		{
		case ActivationType::RELU:
			return ReLU_Backward(grad);
		case ActivationType::SIGMOID:
			return SigmoidBackward(grad);
		case ActivationType::TANH:
			return TanH_Backward(grad);
		case ActivationType::SOFTMAX:
			// Softmax gradient is typically combined with cross-entropy loss, so we can just return the grad here
			return grad;
		default:
			return grad;
		}
	}

private:
	Matrix ReLU(const Matrix& x)
	{
		vector<int> dimensions = x.Dimensions();
		Matrix out(dimensions[0], dimensions[1]);

		vector<vector<double>> matrix = x.GetMatrix();

		for (int row = 0; row < dimensions[0]; row++)
		{
			for (int col = 0; col < dimensions[1]; col++)
			{
				out[row][col] = std::max(0.0, matrix[row][col]);
			}
		}

		return out;
	}

	Matrix ReLU_Backward(const Matrix& grad)
	{
		vector<int> dimensions = grad.Dimensions();
		Matrix out(dimensions[0], dimensions[1]);
		vector<vector<double>> gradMatrix = grad.GetMatrix();
		vector<vector<double>> inputMatrix = lastInput.GetMatrix();
		for (int row = 0; row < dimensions[0]; row++)
		{
			for (int col = 0; col < dimensions[1]; col++)
			{
				out[row][col] = inputMatrix[row][col] > 0 ? gradMatrix[row][col] : 0;
			}
		}
		return out;
	}

	Matrix Sigmoid(const Matrix& x)
	{
		vector<int> dimensions = x.Dimensions();
		Matrix out(dimensions[0], dimensions[1]);

		vector<vector<double>> matrix = x.GetMatrix();

		for (int row = 0; row < dimensions[0]; row++)
		{
			for (int col = 0; col < dimensions[1]; col++)
			{
				out[row][col] = 1.0 / (1.0 + exp(-matrix[row][col]));
			}
		}

		return out;
	}

	Matrix SigmoidBackward(const Matrix& grad)
	{
		vector<int> dimensions = grad.Dimensions();
		Matrix out(dimensions[0], dimensions[1]);
		vector<vector<double>> gradMatrix = grad.GetMatrix();
		vector<vector<double>> outputMatrix = lastOutput.GetMatrix();
		for (int row = 0; row < dimensions[0]; row++)
		{
			for (int col = 0; col < dimensions[1]; col++)
			{
				out[row][col] = gradMatrix[row][col] * outputMatrix[row][col] * (1 - outputMatrix[row][col]);
			}
		}

		return out;
	}

	Matrix TanH(const Matrix& x)
	{
		vector<int> dimensions = x.Dimensions();
		Matrix out(dimensions[0], dimensions[1]);

		vector<vector<double>> matrix = x.GetMatrix();

		for (int row = 0; row < dimensions[0]; row++)
		{
			for (int col = 0; col < dimensions[1]; col++)
			{
				out[row][col] = tanh(matrix[row][col]);
			}
		}

		return out;
	}

	Matrix TanH_Backward(const Matrix& grad)
	{
		vector<int> dimensions = grad.Dimensions();
		Matrix out(dimensions[0], dimensions[1]);
		vector<vector<double>> gradMatrix = grad.GetMatrix();
		vector<vector<double>> outputMatrix = lastOutput.GetMatrix();
		for (int row = 0; row < dimensions[0]; row++)
		{
			for (int col = 0; col < dimensions[1]; col++)
			{
				out[row][col] = gradMatrix[row][col] * (1 - outputMatrix[row][col] * outputMatrix[row][col]);
			}
		}

		return out;
	}

	Matrix Softmax(const Matrix& x)
	{
		vector<int> dimensions = x.Dimensions();
		Matrix out(dimensions[0], dimensions[1]);
		vector<vector<double>> matrix = x.GetMatrix();
		for (int col = 0; col < dimensions[1]; col++)
		{
			double maxVal = matrix[0][col];
			for (int row = 1; row < dimensions[0]; row++)
			{
				if (matrix[row][col] > maxVal)
				{
					maxVal = matrix[row][col];
				}
			}
			double sumExp = 0.0;
			for (int row = 0; row < dimensions[0]; row++)
			{
				sumExp += exp(matrix[row][col] - maxVal);
			}
			for (int row = 0; row < dimensions[0]; row++)
			{
				out[row][col] = exp(matrix[row][col] - maxVal) / sumExp;
			}
		}
		return out;
	}

	Matrix Softmax_Backward(const Matrix& grad)
	{
		return grad;
	}
};

class DenseLayer
{
public:
	Matrix weights;
	Matrix bias;
	Matrix lastInput;
	Activation activation;

	DenseLayer(int inFeatures, int outFeatures, Activation activation)
	{
		weights = Matrix(outFeatures, inFeatures);
		bias = Matrix(outFeatures, 1);
		Matrix::PopulateXavier(weights, inFeatures, outFeatures);
		Matrix::PopulateXavier(bias, outFeatures, 1);
		this->activation = activation;
	}

	Matrix Forward(const Matrix& input)
	{
		lastInput = input;
		Matrix linearOutput = (weights * input) + bias;
		return activation.Forward(linearOutput);
	}

	Matrix Backward(const Matrix& grad, double learningRate)
	{
		Matrix activationGrad = activation.Backward(grad);
		
		Matrix weightGrad = activationGrad * Matrix::Transpose(lastInput);
		Matrix biasGrad = activationGrad;
		
		Matrix outputGrad = Matrix::Transpose(weights) * activationGrad;

		weights = weights - (learningRate * weightGrad);
		bias = bias - (learningRate * biasGrad);
		
		return outputGrad;
	}
};

enum class LossType { MSE, BINARY_CROSS_ENTROPY, CATEGORICAL_CROSS_ENTROPY };

class Loss
{
public: 
	LossType type;
	double lastLoss = 0;

	std::map<int, double> classWeights;
	bool useClassWeights = false;

	Loss() : type(LossType::MSE), lastLoss(0), useClassWeights(false) {}

	Loss(std::string type)
	{
		if (type == "MSE")
		{
			this->type = LossType::MSE;
		}
		else if (type == "BINARY_CROSS_ENTROPY")
		{
			this->type = LossType::BINARY_CROSS_ENTROPY;
		}
		else if (type == "CATEGORICAL_CROSS_ENTROPY")
		{
			this->type = LossType::CATEGORICAL_CROSS_ENTROPY;
		}
		else
		{
			this->type = LossType::MSE;
		}

		useClassWeights = false;
	}

	Loss(std::string type, std::map<int, double> weights)
	{
		if (type == "MSE")
		{
			this->type = LossType::MSE;
		}
		else if (type == "BINARY_CROSS_ENTROPY")
		{
			this->type = LossType::BINARY_CROSS_ENTROPY;
		}
		else if (type == "CATEGORICAL_CROSS_ENTROPY")
		{
			this->type = LossType::CATEGORICAL_CROSS_ENTROPY;
		}
		else
		{
			this->type = LossType::MSE;
		}

		useClassWeights = true;
		classWeights = weights;
	}

	double Forward(const Matrix& predictions, const Matrix& target, int classLabel = -1)
	{
		double loss;

		switch (type)
		{
			case LossType::MSE:
				loss = MSE(predictions, target);
				break;
			case LossType::CATEGORICAL_CROSS_ENTROPY:
				loss = CategoricalCrossEntropy(predictions, target);
				break;
			case LossType::BINARY_CROSS_ENTROPY:
				loss = BinaryCrossEntropy(predictions, target);
				break;
			default:
				loss = MSE(predictions, target);
				break;
		}

		if (useClassWeights && classLabel >= 0 && classWeights.count(classLabel))
		{
			loss *= classWeights[classLabel];
		}

		return loss;
	}

	Matrix Backward(const Matrix& predictions, const Matrix& target, int classLabel = -1)
	{
		Matrix grad(1, 1);
		switch (type)
		{
			case LossType::MSE:
				grad = MSE_Gradient(predictions, target);
				break;
			case LossType::CATEGORICAL_CROSS_ENTROPY:
				grad = CategoricalCrossEntropy_Gradient(predictions, target);
				break;
			case LossType::BINARY_CROSS_ENTROPY:
				grad = BinaryCrossEntropy_Gradient(predictions, target);
				break;
			default:
				grad = MSE_Gradient(predictions, target);
				break;
		}

		if (useClassWeights && classLabel >= 0 && classWeights.count(classLabel))
		{
			double weight = classWeights[classLabel];
			for (int row = 0; row < grad.Dimensions()[0]; row++)
			{
				grad[row][0] *= weight;
			}
		}

		return grad;
	}

private:
	const double EPSILON = 1e-7;

	double MSE(const Matrix& predictions, const Matrix& target)
	{
		vector<int> dimensions = predictions.Dimensions();
		double sum = 0.0;
		for (int row = 0; row < dimensions[0]; row++)
		{
			for (int col = 0; col < dimensions[1]; col++)
			{
				double diff = predictions[row][col] - target[row][col];
				sum += diff * diff;
			}
		}
		lastLoss = sum / (dimensions[0] * dimensions[1]);
		return lastLoss;
	}

	Matrix MSE_Gradient(const Matrix& predictions, const Matrix& target)
	{
		vector<int> dimensions = predictions.Dimensions();
		Matrix grad(dimensions[0], dimensions[1]);
		for (int row = 0; row < dimensions[0]; row++)
		{
			for (int col = 0; col < dimensions[1]; col++)
			{
				grad[row][col] = 2.0 * (predictions[row][col] - target[row][col]) / (dimensions[0] * dimensions[1]);
			}
		}
		return grad;
	}

	double BinaryCrossEntropy(const Matrix& predictions, const Matrix& target)
	{
		vector<int> dimensions = predictions.Dimensions();
		double sum = 0.0;

		for (int row = 0; row < dimensions[0]; row++)
		{
			for (int col = 0; col < dimensions[1]; col++)
			{
				double pred = predictions[row][col];
				pred = std::max(std::min(pred, 1.0 - EPSILON), EPSILON);
				double targ = target[row][col];
				sum += -targ * log(pred) - (1 - targ) * log(1 - pred);
			}
		}
		lastLoss = sum / (dimensions[0] * dimensions[1]);
		return lastLoss;
	}

	Matrix BinaryCrossEntropy_Gradient(const Matrix& predictions, const Matrix& target)
	{
		vector<int> dimensions = predictions.Dimensions();
		Matrix grad(dimensions[0], dimensions[1]);

		for (int row = 0; row < dimensions[0]; row++)
		{
			for (int col = 0; col < dimensions[1]; col++)
			{
				double pred = predictions[row][col];
				pred = std::max(std::min(pred, 1.0 - EPSILON), EPSILON);
				double targ = target[row][col];
				grad[row][col] = (pred - targ) / (dimensions[0] * dimensions[1]);
			}
		}
		return grad;
	}

	double CategoricalCrossEntropy(const Matrix& predictions, const Matrix& target)
	{
		vector<int> dimensions = predictions.Dimensions();
		double sum = 0.0;
		for (int row = 0; row < dimensions[0]; row++)
		{
			for (int col = 0; col < dimensions[1]; col++)
			{
				double pred = predictions[row][col];
				pred = std::max(std::min(pred, 1.0 - EPSILON), EPSILON);
				double targ = target[row][col];
				sum += -targ * log(pred);
			}
		}
		lastLoss = sum / (dimensions[0] * dimensions[1]);
		return lastLoss;
	}

	Matrix CategoricalCrossEntropy_Gradient(const Matrix& predictions, const Matrix& target)
	{
		vector<int> dimensions = predictions.Dimensions();
		Matrix grad(dimensions[0], dimensions[1]);

		for (int row = 0; row < dimensions[0]; row++)
		{
			for (int col = 0; col < dimensions[1]; col++)
			{
				grad[row][col] = (predictions[row][col] - target[row][col]) / (dimensions[0] * dimensions[1]);
			}
		}
		return grad;
	}
};

class NeuralNetwork
{
public:
	NeuralNetwork()
	{
	} 
};