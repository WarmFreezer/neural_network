#include "Tensor.h"
#include "Matrix.h"

#include <vector>
#include <iostream>

Tensor::Tensor(float data)
{
	this->data.push_back(data);
}

Tensor::Tensor(std::vector<float> data)
{
	this->data = data;
	this->shape.push_back(data.size());
	this->stride.push_back(1);
}

Tensor::Tensor(std::vector<std::vector<float>> data)
{
	this->data = Matrix::Flat(data);
	this->shape.push_back(data.size());
	this->shape.push_back(data[0].size());
	this->stride.push_back(data[0].size());
	this->stride.push_back(1);
}