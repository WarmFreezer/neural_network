#pragma once
#include <iostream>
#include <vector>
#include <omp.h>

#include "Matrix.h"

#define Matrix matrix::Matrix

enum class ActivationType { RELU, SIGMOID, TANH, None };

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
	double lastLoss;

	Loss() : type(LossType::MSE), lastLoss(0) {}

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
	}

	double Forward(const Matrix& predictions, const Matrix& target)
	{
		switch (type)
		{
			case LossType::MSE:
				return MSE(predictions, target);
			case LossType::CATEGORICAL_CROSS_ENTROPY:
				return CategoricalCrossEntropy(predictions, target);
			case LossType::BINARY_CROSS_ENTROPY:
				return BinaryCrossEntropy(predictions, target);
			default:
				return MSE(predictions, target);
		}
	}

	Matrix Backward(const Matrix& predictions, const Matrix& target)
	{
		switch (type)
		{
			case LossType::MSE:
				return MSE_Gradient(predictions, target);
			case LossType::CATEGORICAL_CROSS_ENTROPY:
				return CategoricalCrossEntropy_Gradient(predictions, target);
			case LossType::BINARY_CROSS_ENTROPY:
				return BinaryCrossEntropy_Gradient(predictions, target);
			default:
				return MSE_Gradient(predictions, target);
		}
	}

private:
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
		const double epsilon = 1e-15; // To prevent log(0)
		for (int row = 0; row < dimensions[0]; row++)
		{
			for (int col = 0; col < dimensions[1]; col++)
			{
				double pred = predictions[row][col];
				pred = std::max(std::min(pred, 1.0 - epsilon), epsilon);
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
		const double epsilon = 1e-15; // To prevent division by zero
		for (int row = 0; row < dimensions[0]; row++)
		{
			for (int col = 0; col < dimensions[1]; col++)
			{
				double pred = predictions[row][col];
				pred = std::max(std::min(pred, 1.0 - epsilon), epsilon);
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
		const double epsilon = 1e-15; // To prevent log(0)
		for (int row = 0; row < dimensions[0]; row++)
		{
			for (int col = 0; col < dimensions[1]; col++)
			{
				double pred = predictions[row][col];
				pred = std::max(std::min(pred, 1.0 - epsilon), epsilon);
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
		const double epsilon = 1e-15; // To prevent division by zero
		for (int row = 0; row < dimensions[0]; row++)
		{
			for (int col = 0; col < dimensions[1]; col++)
			{
				double pred = predictions[row][col];
				pred = std::max(std::min(pred, 1.0 - epsilon), epsilon);
				double targ = target[row][col];
				grad[row][col] = -targ / pred / (dimensions[0] * dimensions[1]);
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