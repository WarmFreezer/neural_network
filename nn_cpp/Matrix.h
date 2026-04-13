#pragma once
#include <iostream>
#include <vector>
#include <random>
#include <string>
#include <omp.h>

#define vector std::vector

namespace matrix
{
	enum class Op { ADD, SUB, MUL, NONE };

	class Matrix
	{
	public:
		Matrix();
		Matrix(int x, int y);
		Matrix(int x, int y, Matrix* parents[2], Op op);

		static void PrintMatrix(const Matrix& a);
		static void PopulateRand(Matrix& a);
		static void PopulateXavier(Matrix& a, int fanIn, int fanOut);
		static void SetNumThreads(int numThreads);

		Matrix operator+(const Matrix& other);
		Matrix operator+(double val) const;
		friend Matrix operator+(double val, const Matrix& matrix);
		Matrix operator-(const Matrix& other);
		Matrix operator-(double val) const;
		friend Matrix operator-(double val, const Matrix& matrix);
		Matrix operator*(const Matrix& other);
		Matrix operator*(double other);
		friend Matrix operator*(double val, const Matrix& other);
		Matrix operator%(const Matrix& other);
		Matrix operator=(Matrix other);
		const vector<double>& operator[] (int index) const;
		vector<double>& operator[] (int index);

		vector<int> Dimensions() const;
		void Backward(const Matrix& upstreamGrad);
		vector<vector<double>> GetGrad() const;
		vector<vector<double>> GetMatrix() const;
		void ZeroGrad();

		template <typename T>
		static vector<T> Flatten(const vector<vector<T>> v);

		template <typename T>
		static vector<vector<T>> Gridify(const vector<T> v, int n, int k);
			
		static Matrix Transpose(const Matrix& matrix);

	private:
		vector<vector<double>> matrix;
		vector<vector<double>> grad;
		Matrix* parents[2];
		Op operation = Op::NONE;
	};
}