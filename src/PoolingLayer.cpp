#include "PoolingLayer.h"

PoolingLayer::PoolingLayer(int pool_size, int stride, PoolingType type) : pool_size_(pool_size), stride_(stride), type_(type), input_height_(0), input_width_(0), input_channels_(0) {}

int PoolingLayer::output_size(int input_size, int kernel, int stride) {
    return (input_size - kernel) / stride + 1;
}

string PoolingLayer::getTypeName() const {
    switch (type_) {
        case PoolingType::Max: return "MaxPool";
        case PoolingType::Min: return "MinPool";
        case PoolingType::Average: return "AvgPool";
        default: return "Unknown";
    }
}

Tensor3D PoolingLayer::forward(const Tensor3D& input) {
    input_height_ = input.height;
    input_width_ = input.width;
    input_channels_ = input.channels;

    int output_height = output_size(input_height_, pool_size_, stride_);
    int output_width = output_size(input_width_, pool_size_, stride_);

    if (output_height <= 0 || output_width <= 0) {
        throw invalid_argument("Invalid pooling parameters: output dimensions must be positive.");
    }

    Tensor3D output(input.channels, output_height, output_width, 0.0);
    selected_indices_.clear();
    selected_indices_.resize(input.channels * output_height * output_width, -1);

    for (int c = 0; c < input.channels; c++) {
        for (int oh = 0; oh < output_height; oh++) {
            for (int ow = 0; ow < output_width; ow++) {
                int output_index = c * output_height * output_width + oh * output_width + ow;

                if (type_ == PoolingType::Average) {
                    double sum = 0.0;
                    for (int ph = 0; ph < pool_size_; ph++) {
                        for (int pw = 0; pw < pool_size_; pw++) {
                            int ih = oh * stride_ + ph;
                            int iw = ow * stride_ + pw;
                            sum += input.at(c, ih, iw);
                        }
                    }

                    output.at(c, oh, ow) = sum / (pool_size_ * pool_size_);
                }
                else {
                    bool isMax = (type_ == PoolingType::Max);
                    double best_value = isMax ? -numeric_limits<double>::infinity() : numeric_limits<double>::infinity();
                    int best_ih = 0, best_iw = 0;

                    for (int ph = 0; ph < pool_size_; ph++) {
                        for (int pw = 0; pw < pool_size_; pw++) {
                            int ih = oh * stride_ + ph;
                            int iw = ow * stride_ + pw;
                            double val = input.at(c, ih, iw);

                            bool is_better = isMax ? (val > best_value) : (val < best_value);
                            if (is_better) {
                                best_value = val;
                                best_ih = ih;
                                best_iw = iw;
                            }
                        }
                    }
                    output.at(c, oh, ow) = best_value;
                    selected_indices_[output_index] = c * input_height_ * input_width_ + best_ih * input_width_ + best_iw;
                }
            }
        }
    }

    return output;
}

Tensor3D PoolingLayer::backward(const Tensor3D& gradient_output) const {
    Tensor3D gradient_input(input_channels_, input_height_, input_width_, 0.0);

    int output_height = gradient_output.height;
    int output_width = gradient_output.width;

    for (int c = 0; c < gradient_output.channels; c++) {
        for (int oh = 0; oh < output_height; oh++) {
            for (int ow = 0; ow < output_width; ow++) {
                double gradient_value = gradient_output.at(c, oh, ow);
                int output_index = c * output_height * output_width + oh * output_width + ow;

                if (type_ == PoolingType::Average) {
                    double distributed = gradient_value / (pool_size_ * pool_size_);
                    for (int ph = 0; ph < pool_size_; ph++) {
                        for (int pw = 0; pw < pool_size_; pw++) {
                            int ih = oh * stride_ + ph;
                            int iw = ow * stride_ + pw;
                            gradient_input.at(c, ih, iw) += distributed;
                        }
                    }
                }
                else {
                    int selected_index = selected_indices_[output_index];
                    gradient_input.data[selected_index] += gradient_value;
                }
            }
        }
    }

    return gradient_input;
}