#include "Matrix.h"

#include <iostream>
#include <vector>
#include <random>
#include <string>
#include <omp.h>

using namespace std;
namespace matrix
{
	Matrix::Matrix(int x, int y)
	{
		if (x <= 0 || y <= 0)
		{
			cout << "Invalid matrix dimensions.\n";
			throw std::errc::invalid_argument;
		}

		matrix = vector<vector<double>>(x, vector<double>(y));
	}

	void Matrix::PrintMatrix(const Matrix a)
	{
		//Prints each value in the matrix
		for (vector<double> i : a.matrix)
		{
			for (double j : i)
			{
				cout << j << " ";
			}
			cout << endl;
		}
	}

	void Matrix::PopulateRand(Matrix& a)
	{
		mt19937 rng(std::random_device{}());
		uniform_real_distribution<double> dist(-10000.0, 10000.0);

		//Populates the matrix's matrix with random integers from -1000 to 1000
		for (vector<double>& i : a.matrix)
		{
			for (double& j : i)
			{
				j = dist(rng);
			}
		}
	}

	void Matrix::SetNumThreads(int numThreads)
	{
		//Used for evaluating the most optimal thread count for multiplication
		omp_set_num_threads(numThreads);
	}

	Matrix Matrix::operator+(Matrix matrix2)
	{
		if (this->matrix.size() == matrix2.matrix.size() && this->matrix[0].size() == matrix2.matrix.size())
		{
			int n = this->matrix.size();
			int k = this->matrix[0].size();
			Matrix sum(n, k);

			for (int row = 0; row < n; row++)
			{
				for (int col = 0; col < k; col++)
				{
					sum[row][col] = this->matrix[row][col] + matrix2.matrix[row][col];
				}
			}

			return sum;
		}
		else if (this->matrix.size() == 1 && this->matrix[0].size() == 1)
		{
			return this->matrix[0][0] + matrix2;
		}
		else if (matrix2.matrix.size() == 1 && matrix2.matrix[0].size() == 1)
		{
			return matrix2[0][0] + *this;
		}
		else
		{
			cout << "Cannot add different sized matrices.\n";
			throw std::errc::invalid_argument;
		}
	}

	Matrix Matrix::operator+(double val)
	{
		Matrix sum = Matrix(this->matrix.size(), this->matrix[0].size());
		for (int row = 0; row < this->matrix.size(); row++)
		{
			for (int col = 0; col < this->matrix[0].size(); col++)
			{
				sum[row][col] = this->matrix[row][col] + val;
			}
		}
		return sum;
	}

	Matrix operator+(double val, Matrix& matrix)
	{
		Matrix sum = Matrix(matrix.matrix.size(), matrix.matrix[0].size());
		for (int row = 0; row < matrix.matrix.size(); row++)
		{
			for (int col = 0; col < matrix.matrix[0].size(); col++)
			{
				sum[row][col] = matrix.matrix[row][col] + val;
			}
		}
		return sum;
	}

	Matrix Matrix::operator*(Matrix matrix2)
	{
		if (this->matrix[0].size() == matrix2.matrix.size())
		{
			int n = this->matrix.size();
			int k = this->matrix[0].size();
			int m = matrix2.matrix[0].size();

			Matrix productMatrix(n, m);

#pragma omp parallel for
			for (int row = 0; row < n; row++)
			{
				for (int col = 0; col < m; col++)
				{
					//Calculate the value based on matrix multiplication rules
					double indexVal = 0;
					for (int ki = 0; ki < k; ki++)
					{
						indexVal += this->matrix[row][ki] * matrix2.matrix[ki][col];
					}

					//Set the value after the for loop to potentially reduce false sharing
					productMatrix[row][col] = indexVal;
				}
			}

			return productMatrix;
		}
		else
		{
			cout << "Cannot multiply different sized matrices.\n";
			throw std::errc::invalid_argument;
		}
	}

	vector<double>& Matrix::operator[] (int index)
	{
		if (index < 0 || index >= this->matrix.size()) {
			throw std::out_of_range("Index out of bounds");
		}

		return this->matrix[index];
	}

	vector<int> Matrix::Dimensions()
	{
		vector<int> dimensions;
		dimensions.push_back(this->matrix.size()); //Rows
		dimensions.push_back(this->matrix[0].size()); //Cols
		return dimensions;
	}

	template <typename T>
	vector<T> Matrix::Flatten(const vector<vector<T>> v)
	{
		vector<T> flat;
		for (int row = 0; row < v.size(); row++)
		{
			for (int col = 0; col < v[0].size(); col++)
			{
				flat.push_back(v[row][col]);
			}
		}

		return flat;
	}

	template <typename T>
	vector<vector<T>> Matrix::Gridify(const vector<T> v, int n, int k)
	{
		vector<vector<T>> result(n, vector<T>(k));

		for (int row = 0; row < n; row++)
		{
			for (int col = 0; col < k; col++)
			{
				result[row][col] = v[row * k + col];
			}
		}

		return result;
	}
}
