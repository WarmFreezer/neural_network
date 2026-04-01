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
		Matrix::PopulateRand(weights);
		Matrix::PopulateRand(bias);
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
		
		weights = weights - (learningRate * weightGrad);
		bias = bias - (learningRate * biasGrad);
		
		return Matrix::Transpose(weights) * activationGrad;
	}
};

class NeuralNetwork
{
public:
	NeuralNetwork()
	{
	}
};