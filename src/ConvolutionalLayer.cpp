#include "ConvolutionalLayer.h"
#include <cmath>
#include <stdexcept>

ConvolutionalLayer::ConvolutionalLayer(int input_channels, int output_channels, int kernel, int stride, int padding, int seed) : input_channels_(input_channels), output_channels_(output_channels), kernel_(kernel), stride_(stride), padding_(padding) {
    mt19937 rng(seed);
    double scale = sqrt(2.0 / (input_channels * kernel * kernel));

    kernels_.reserve(output_channels);
    biases_.reserve(output_channels);

    for (int i = 0; i < output_channels; i++) {
        kernels_.push_back(tensorRandom(input_channels, kernel, kernel, scale, rng));
    }
}

int ConvolutionalLayer::size_out(int input_size, int kernel, int stride, int padding) {
    return (input_size - kernel + 2 * padding) / stride + 1;
}

Tensor3D ConvolutionalLayer::apply_padding(const Tensor3D& input) const {
    if (padding_ == 0) {
        return input;
    }

    int new_height = input.height + 2 * padding_;
    int new_width = input.width + 2 * padding_;
    Tensor3D padded(input.channels, new_height, new_width, 0.0);

    for (int c = 0; c < input.channels; c++) {
        for (int h = 0; h < input.height; h++) {
            for (int w = 0; w < input.width; w++) {
                padded.at(c, h + padding_, w + padding_) = input.at(c, h, w);
            }
        }
    }

    return padded;
}

Tensor3D ConvolutionalLayer::forward(const Tensor3D& input) {
    if (input.channels != input_channels_) {
        throw invalid_argument("Input channels do not match layer configuration.");
    }

    input_height_ = input.height;
    input_width_ = input.width;
    input_cache_ = apply_padding(input);

    int output_height = size_out(input_cache_.height, kernel_, stride_, padding_);
    int output_width = size_out(input_cache_.width, kernel_, stride_, padding_);
    Tensor3D output(output_channels_, output_height, output_width);

    for (int f = 0; f < output_channels_; f++) {
        for (int h = 0; h < output_height; h++) {
            for (int w = 0; w < output_width; w++) {
                double sum = biases_[f];
                for (int c = 0; c < input_channels_; c++) {
                    for (int kh = 0; kh < kernel_; kh++) {
                        for (int kw = 0; kw < kernel_; kw++) {
                            int in_h = h * stride_ + kh;
                            int in_w = w * stride_ + kw;
                            sum += input_cache_.at(c, in_h, in_w) * kernels_[f].at(c, kh, kw);
                        }
                    }
                }
                output.at(f, h, w) = sum;
            }
        }
    }

    return output;
}

Tensor3D ConvolutionalLayer::backward(const Tensor3D& gradient_output, double learning_rate) {
    int output_height = gradient_output.height;
    int output_width = gradient_output.width;

    vector<Tensor3D> gradient_kernels(output_channels_, Tensor3D(input_channels_, kernel_, kernel_, 0.0));
    vector<double> gradient_biases(output_channels_, 0.0);
    Tensor3D gradient_padded(input_channels_, input_cache_.height, input_cache_.width, 0.0);

    for (int f = 0; f < output_channels_; f++) {
        for (int h = 0; h < output_height; h++) {
            for (int w = 0; w < output_width; w++) {
                double grad = gradient_output.at(f, h, w);
                gradient_biases[f] += grad;
                for (int c = 0; c < input_channels_; c++) {
                    for (int kh = 0; kh < kernel_; kh++) {
                        for (int kw = 0; kw < kernel_; kw++) {
                            int in_h = h * stride_ + kh;
                            int in_w = w * stride_ + kw;
                            
                            gradient_kernels[f].at(c, kh, kw) += grad * input_cache_.at(c, in_h, in_w);
                            gradient_padded.at(c, in_h, in_w) += grad * kernels_[f].at(c, kh, kw);
                        }
                    }
                }
            }
        }
    }

    for (int f = 0; f < output_channels_; f++) {
        biases_[f] -= learning_rate * gradient_biases[f];
        for (int i = 0; i < gradient_kernels[f].size(); i++) {
            kernels_[f].data[i] -= learning_rate * gradient_kernels[f].data[i];
        }
    }

    if (padding_ == 0) {
        return gradient_padded;
    }

    Tensor3D gradient_input(input_channels_, input_height_, input_width_, 0.0);
    for (int c = 0; c < input_channels_; c++) {
        for (int h = 0; h < input_height_; h++) {
            for (int w = 0; w < input_width_; w++) {
                gradient_input.at(c, h, w) = gradient_padded.at(c, h + padding_, w + padding_);
            }
        }
    }

    return gradient_input;
}