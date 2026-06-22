#pragma once

#include "Tensor3D.h"
#include <vector>
#include <string>
#include <limits>
#include <stdexcept>

using namespace std;

enum class PoolingType { Max, Min, Average };

class PoolingLayer {
    public:
        PoolingLayer(int kernel, int stride, PoolingType type);
        Tensor3D forward(const Tensor3D& input);
        Tensor3D backward(const Tensor3D& gradient_output) const;

        static int output_size(int input_size, int kernel, int stride);
        string getTypeName() const;
    
    private:
        int pool_size_, stride_;
        PoolingType type_;

        vector<int> selected_indices_;
        int input_height_, input_width_, input_channels_;
};