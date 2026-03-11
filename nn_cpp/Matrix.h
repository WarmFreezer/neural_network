#pragma once
#include <iostream>
#include <vector>
#include <random>
#include <string>
#include <omp.h>

#define vector std::vector

namespace matrix
{
	class Matrix
	{
	public:
		Matrix(int x, int y);

		Matrix(int x, int y, Matrix* parents[2]);

		static void PrintMatrix(const Matrix a);

		static void PopulateRand(Matrix& a);

		static void SetNumThreads(int numThreads);

		Matrix operator+(Matrix matrix2);

		Matrix operator+(double val);

		friend Matrix operator+(double val, Matrix& matrix);

		Matrix operator*(Matrix matrix2);

		vector<double>& operator[] (int index);

		vector<int> Dimensions();

		template <typename T>
		static vector<T> Flatten(const vector<vector<T>> v);

		template <typename T>
		static vector<vector<T>> Gridify(const vector<T> v, int n, int k);

	private:
		vector<vector<double>> matrix;

		// Used for backpropagation to track the parents of the matrix
		Matrix* parents[2];
	};
}