#include <iostream>
#include <cassert>

#include "nn.cpp"
#include "Matrix.h"

using namespace std;
using namespace matrix;

void TestMatrixAddition()
{
	Matrix a(2, 2);
	Matrix b(2, 2);
	a[0][0] = 1; a[0][1] = 2;
	a[1][0] = 3; a[1][1] = 4;
	b[0][0] = 5; b[0][1] = 6;
	b[1][0] = 7; b[1][1] = 8;

	Matrix c = a + b;
	assert(c[0][0] == 6 && c[0][1] == 8);
	cout << "✓ Matrix Addition Test Passed\n";
}

void TestMatrixMultiplication()
{
	Matrix a(2, 3);
	Matrix b(3, 2);

	// Fill with known values
	for (int i = 0; i < 2; i++)
		for (int j = 0; j < 3; j++)
			a[i][j] = 1;
	for (int i = 0; i < 3; i++)
		for (int j = 0; j < 2; j++)
			b[i][j] = 1;

	Matrix c = a * b;
	assert(c[0][0] == 3 && c[1][1] == 3);
	cout << "✓ Matrix Multiplication Test Passed\n";
}

void TestMatrixTranspose()
{
	Matrix a(2, 3);
	a[0][0] = 1; a[0][1] = 2; a[0][2] = 3;
	a[1][0] = 4; a[1][1] = 5; a[1][2] = 6;

	Matrix t = Matrix::Transpose(a);
	assert(t.Dimensions()[0] == 3 && t.Dimensions()[1] == 2);
	assert(t[0][0] == 1 && t[1][0] == 2 && t[2][0] == 3);
	cout << "✓ Matrix Transpose Test Passed\n";
}

void TestLinearLayerForward()
{
	LinearLayer layer(3, 2);
	Matrix input(3, 1);
	input[0][0] = 1; input[1][0] = 2; input[2][0] = 3;

	Matrix output = layer.Forward(input);
	assert(output.Dimensions()[0] == 2 && output.Dimensions()[1] == 1);
	cout << "✓ Linear Layer Forward Test Passed\n";
}

void TestActivationFunctions()
{
	Matrix input(2, 2);
	input[0][0] = 0;    input[0][1] = 1;
	input[1][0] = -1;   input[1][1] = 2;

	Activation relu("RELU");
	Matrix reluOut = relu.Forward(input);
	assert(reluOut[0][0] == 0 && reluOut[0][1] == 1);
	assert(reluOut[1][0] == 0 && reluOut[1][1] == 2);
	cout << "✓ ReLU Activation Test Passed\n";

	Activation sigmoid("SIGMOID");
	Matrix sigmoidOut = sigmoid.Forward(input);
	assert(sigmoidOut[0][0] > 0 && sigmoidOut[0][0] < 1);
	cout << "✓ Sigmoid Activation Test Passed\n";
}

void NumericalGradientCheck()
{
	Matrix a(2, 2);
	Matrix b(2, 2);
	Matrix::PopulateRand(a);
	Matrix::PopulateRand(b);

	double epsilon = 1e-5;

	// Numerical gradient for a[0][0]
	a[0][0] += epsilon;
	Matrix f_plus = a * b;
	a[0][0] -= 2 * epsilon;
	Matrix f_minus = a * b;

	double numericalGrad = (f_plus[0][0] - f_minus[0][0]) / (2 * epsilon);
	cout << "Numerical Gradient: " << numericalGrad << "\n";
	cout << "✓ Gradient Checking Setup Complete\n";
}

void TestSimpleNetwork()
{
	cout << "\n=== Simple Network Test ===\n";

	// Create layers
	Activation relu("RELU");
	DenseLayer layer1(2, 4, relu);

	Activation none("NONE");
	DenseLayer layer2(4, 1, none);

	// Create dummy input (batch_size=1, features=2)
	Matrix input(2, 1);
	input[0][0] = 0.5;
	input[1][0] = -0.3;

	// Forward pass
	Matrix hidden = layer1.Forward(input);
	Matrix output = layer2.Forward(hidden);

	cout << "Input dims: " << input.Dimensions()[0] << "x" << input.Dimensions()[1] << "\n";
	cout << "Output dims: " << output.Dimensions()[0] << "x" << output.Dimensions()[1] << "\n";
	cout << "✓ Simple Network Forward Pass Complete\n";

	// Backward pass (dummy gradient)
	Matrix lossGrad(1, 1);
	lossGrad[0][0] = 0.1;

	Matrix grad2 = layer2.Backward(lossGrad, 0.01);
	Matrix grad1 = layer1.Backward(grad2, 0.01);

	cout << "✓ Simple Network Backward Pass Complete\n";
}

int main()
{
	cout << "=== Matrix Operation Tests ===\n";
	TestMatrixAddition();
	TestMatrixMultiplication();
	TestMatrixTranspose();

	cout << "\n=== Layer Forward/Backward Tests ===\n";
	TestLinearLayerForward();
	TestActivationFunctions();
	
	cout << "\n=== Gradient Checking ===\n";
	NumericalGradientCheck();

	TestSimpleNetwork();

	return 0;
}