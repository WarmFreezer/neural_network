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
		~Matrix();
		Matrix();
		Matrix(int x, int y);
		Matrix(const Matrix& other);
		Matrix(Matrix&& other) noexcept;

		static void PrintMatrix(const Matrix& a);
		static void PopulateRand(Matrix& a);
		static void PopulateXavier(Matrix& a, int fanIn, int fanOut);

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
		Matrix& operator=(const double* other);
		Matrix& operator=(const Matrix& other);
		Matrix& operator=(Matrix&& other) noexcept;
		const double* operator[] (int index) const;
		double* operator[] (int index);

		int Rows() const;
		int Cols() const;
		vector<int> Dimensions() const;
		double* GetMatrix() const;

		template <typename T>
		static vector<T> ToArray(const Matrix v);

		template <typename T>
		static vector<vector<T>> Gridify(const vector<T> v, int n, int k);
			
		static Matrix Transpose(const Matrix& matrix);

	private:
		int dimensions[2];
		double* matrix;
	};
}