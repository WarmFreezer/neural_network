#include "Matrix.h"

#include <iostream>
#include <vector>
#include <random>
#include <string>
#include <omp.h>

using namespace std;
namespace matrix
{
	Matrix::Matrix()
	{
		matrix = vector<vector<double>>();
		parents[0] = nullptr;
		parents[1] = nullptr;
	}

	Matrix::Matrix(int x, int y)
	{
		if (x <= 0 || y <= 0)
		{
			cout << "Invalid matrix dimensions.\n";
			throw std::errc::invalid_argument;
		}

		matrix = vector<vector<double>>(x, vector<double>(y));
		parents[0] = nullptr;
		parents[1] = nullptr;
	}

	Matrix::Matrix(int x, int y, Matrix* parents[2], Op op)
	{
		if (x <= 0 || y <= 0)
		{
			cout << "Invalid matrix dimensions.\n";
			throw std::errc::invalid_argument;
		}

		matrix = vector<vector<double>>(x, vector<double>(y));
		this->parents[0] = parents[0];
		this->parents[1] = parents[1];
		operation = op;
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

	void Matrix::PopulateXavier(Matrix& a, int fanIn, int fanOut)
	{
		mt19937 rng(std::random_device{}());
		double limit = sqrt(6.0 / (a.matrix.size() + a.matrix[0].size()));
		uniform_real_distribution<double> dist(-limit, limit);

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

	Matrix Matrix::operator+(const Matrix& other)
	{
		Matrix* parents[2] = { this, const_cast<Matrix*>(&other) };
		if (this->matrix.size() == other.matrix.size() && this->matrix[0].size() == other.matrix[0].size())
		{
			Matrix sum(this->matrix.size(), this->matrix[0].size(), parents, Op::ADD);

			for (int row = 0; row < this->matrix.size(); row++)
			{
				for (int col = 0; col < this->matrix[0].size(); col++)
				{
					sum[row][col] = this->matrix[row][col] + other.matrix[row][col];
				}
			}

			return sum;
		}
		else if (this->matrix.size() == 1 && this->matrix[0].size() == 1)
		{
			Matrix returnMe = this->matrix[0][0] + other;
			returnMe.parents[0] = parents[0];
			returnMe.parents[1] = parents[1];

			return returnMe;
		}
		else if (other.matrix.size() == 1 && other.matrix[0].size() == 1)
		{
			Matrix returnMe = other.matrix[0][0] + *this;
			returnMe.parents[0] = parents[0];
			returnMe.parents[1] = parents[1];

			return returnMe;
		}
		else
		{
			cout << "Cannot add different sized matrices.\n";
			throw std::errc::invalid_argument;
		}
	}

	Matrix Matrix::operator+(double val) const
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

	Matrix operator+(double val, const Matrix& matrix)
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

	Matrix Matrix::operator-(const Matrix& other)
	{
		Matrix* parents[2] = { this, const_cast<Matrix*>(&other)};
		if (this->matrix.size() == other.matrix.size() && this->matrix[0].size() == other.matrix[0].size())
		{
			Matrix diff(this->matrix.size(), this->matrix[0].size(), parents, Op::SUB);

			for (int row = 0; row < this->matrix.size(); row++)
			{
				for (int col = 0; col < this->matrix[0].size(); col++)
				{
					diff[row][col] = this->matrix[row][col] - other.matrix[row][col];
				}
			}

			return diff;
		}
		else if (this->matrix.size() == 1 && this->matrix[0].size() == 1)
		{
			Matrix returnMe = this->matrix[0][0] - other;
			returnMe.parents[0] = parents[0];
			returnMe.parents[1] = parents[1];

			return returnMe;
		}
		else if (other.matrix.size() == 1 && other.matrix[0].size() == 1)
		{
			Matrix returnMe = other.matrix[0][0] - *this;
			returnMe.parents[0] = parents[0];
			returnMe.parents[1] = parents[1];

			return returnMe;
		}
		else
		{
			cout << "Cannot add different sized matrices.\n";
			throw std::errc::invalid_argument;
		}
	}

	Matrix Matrix::operator-(double val) const
	{
		Matrix diff = Matrix(this->matrix.size(), this->matrix[0].size());
		for (int row = 0; row < this->matrix.size(); row++)
		{
			for (int col = 0; col < this->matrix[0].size(); col++)
			{
				diff[row][col] = this->matrix[row][col] - val;
			}
		}
		return diff;
	}

	Matrix operator-(double val, const Matrix& matrix)
	{
		Matrix diff = Matrix(matrix.GetMatrix().size(), matrix.GetMatrix()[0].size());
		for (int row = 0; row < matrix.GetMatrix().size(); row++)
		{
			for (int col = 0; col < matrix.GetMatrix()[0].size(); col++)
			{
				diff[row][col] = val - matrix.GetMatrix()[row][col];
			}
		}
		return diff;
	}

	Matrix Matrix::operator*(const Matrix& other)
	{
		if (this->matrix[0].size() == other.matrix.size())
		{
			int n = this->matrix.size();
			int k = this->matrix[0].size();
			int m = other.matrix[0].size();

			Matrix* parents[2] = { this, const_cast<Matrix*>(&other)};

			Matrix productMatrix(n, m, parents, Op::MUL);

#pragma omp parallel for
			for (int row = 0; row < n; row++)
			{
				for (int col = 0; col < m; col++)
				{
					//Calculate the value based on matrix multiplication rules
					double indexVal = 0;
					for (int ki = 0; ki < k; ki++)
					{
						indexVal += this->matrix[row][ki] * other.matrix[ki][col];
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

	Matrix Matrix::operator*(double other)
	{
		Matrix productMatrix = Matrix(this->matrix.size(), this->matrix[0].size());
		for (int row = 0; row < this->matrix.size(); row++)
		{
			for (int col = 0; col < this->matrix[0].size(); col++)
			{
				productMatrix[row][col] = this->matrix[row][col] * other;
			}
		}
		return productMatrix;
	}

	Matrix operator*(double val, Matrix other)
	{
		Matrix productMatrix = Matrix(other.matrix.size(), other.matrix[0].size());
		for (int row = 0; row < other.matrix.size(); row++)
		{
			for (int col = 0; col < other.matrix[0].size(); col++)
			{
				productMatrix[row][col] = other.matrix[row][col] * val;
			}
		}
		return productMatrix;
	}

	Matrix Matrix::operator%(Matrix other)
	{
		if (this->matrix.size() == other.matrix.size() && this->matrix[0].size() == other.matrix[0].size())
		{
			Matrix productMatrix(this->matrix.size(), this->matrix[0].size());
			for (int row = 0; row < this->matrix.size(); row++)
			{
				for (int col = 0; col < this->matrix[0].size(); col++)
				{
					productMatrix[row][col] = this->matrix[row][col] * other.matrix[row][col];
				}
			}
			return productMatrix;
		}
		else
		{
			throw std::errc::invalid_argument;
		}
	}

	Matrix Matrix::operator=(Matrix other)
	{
		this->matrix = other.matrix;
		this->parents[0] = other.parents[0];
		this->parents[1] = other.parents[1];
		return *this;
	}

	vector<double>& Matrix::operator[] (int index)
	{
		if (index < 0 || index >= this->matrix.size()) {
			throw std::out_of_range("Index out of bounds");
		}

		return this->matrix[index];
	}

	void Matrix::Backward(const Matrix& upstreamGrad)
	{
		if (grad.empty())
		{
			grad = upstreamGrad.matrix;
		}
		else
		{
			for (int row = 0; row < grad.size(); row++)
			{
				for (int col = 0; col < grad[0].size(); col++)
				{
					grad[row][col] += upstreamGrad.matrix[row][col];
				}
			}
		}

		switch (operation)
		{
			case Op::ADD:
				if (parents[0] != nullptr)
				{
					parents[0]->Backward(upstreamGrad);
				}
				if (parents[1] != nullptr)
				{
					parents[1]->Backward(upstreamGrad);
				}
				break;
			case Op::SUB:
				if (parents[0] != nullptr)
				{
					parents[0]->Backward(upstreamGrad);
				}
				if (parents[1] != nullptr)
				{
					Matrix negUpstreamGrad(upstreamGrad.matrix.size(), upstreamGrad.matrix[0].size());
					for (int row = 0; row < upstreamGrad.matrix.size(); row++)
					{
						for (int col = 0; col < upstreamGrad.matrix[0].size(); col++)
						{
							negUpstreamGrad[row][col] = -upstreamGrad.matrix[row][col];
						}
					}
					parents[1]->Backward(negUpstreamGrad);
				}
				break;
			case Op::MUL:
				if (parents[0] != nullptr)
				{
					Matrix grad0 = Transpose(upstreamGrad) * (*parents[1]);
					parents[0]->Backward(Transpose(grad0));
				}
				if (parents[1] != nullptr)
				{
					Matrix grad1 = Transpose(*parents[0]) * upstreamGrad;
					parents[1]->Backward(grad1);
				}
				break;
			case Op::NONE:
				break;
		}
	}

	vector<vector<double>> Matrix::GetGrad() const
	{
		return grad;
	}

	vector<vector<double>> Matrix::GetMatrix() const
	{
		return matrix;
	}

	void Matrix::ZeroGrad()
	{
		for (int row = 0; row < grad.size(); row++)
		{
			for (int col = 0; col < grad[0].size(); col++)
			{
				grad[row][col] = 0;
			}
		}
	}

	vector<int> Matrix::Dimensions() const
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

	Matrix Matrix::Transpose(const Matrix& matrix)
	{
		Matrix transposed = Matrix(matrix.matrix[0].size(), matrix.matrix.size());
		for (int row = 0; row < matrix.matrix.size(); row++)
		{
			for (int col = 0; col < matrix.matrix[0].size(); col++)
			{
				transposed[col][row] = matrix.matrix[row][col];
			}
		}
		return transposed;
	}
}
