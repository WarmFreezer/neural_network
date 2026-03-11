#include <iostream>

#include "nn.cpp"
#include "Matrix.h"

using namespace std;
using namespace matrix;

int main()
{
	Matrix a = Matrix(5, 5);
	Matrix b = Matrix(5, 5);
	
	Matrix::PopulateRand(a);
	Matrix::PopulateRand(b);

	Matrix product = a * b;

	Matrix sum = a + 5;

	return 0;
}