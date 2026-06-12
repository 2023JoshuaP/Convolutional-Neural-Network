#pragma once

#include "Tensor3D.h"

class ConvolutionalLayer {
    public:
        ConvolutionalLayer(int input_channels, int output_channels, int kernel, int stride = 1, int padding = 0, int seed = 42);
        Tensor3D forward(const Tensor3D& input);
        Tensor3D backward(const Tensor3D& gradient_output, double learning_rate);

        static int size_out(int input_size, int kernel, int stride, int padding);

        int get_output_channels() const {
            return output_channels_;
        }
        int get_kernel() const {
            return kernel_;
        }
        int get_stride() const {
            return stride_;
        }
        int get_padding() const {
            return padding_;
        }
    
    private:
        int input_channels_, output_channels_, kernel_, stride_, padding_;
        vector<Tensor3D> kernels_;
        vector<double> biases_;
        Tensor3D input_cache_;
        int input_height_, input_width_;
        Tensor3D apply_padding(const Tensor3D& input) const;
};