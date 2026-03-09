#pragma once
#include <vector>

using namespace std;
class Tensor
{
private:
	vector<float> data;
	vector<int> shape;
	vector<int> stride;

public:
	Tensor(float data);
	Tensor(vector<float> data);
	Tensor(vector<vector<float>> data);
};