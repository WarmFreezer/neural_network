#pragma once
#include <vector>

using namespace std;
static class Matrix
{
public:
    template <typename T>
    static vector<T> Flat(const vector<vector<T>> v);

    template <typename T>
    static vector<vector<T>> Grid(const vector<T> v, int n, int k);
}