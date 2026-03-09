#include "Matrix.h"

#include <vector>

using namespace std;
template <typename T>
vector<T> Matrix::Flat(const vector<vector<T>> v)
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
vector<vector<T>> Matrix::Grid(const vector<T> v, int n, int k)
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
