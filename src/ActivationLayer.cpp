#include "ActivationLayer.h"

Tensor3D ReLUActivationLayer::forward(const Tensor3D& input) {
    Tensor3D output(input.channels, input.height, input.width);
    mask_ = Tensor3D(input.channels, input.height, input.width, 0.0);

    for (int i = 0; i < input.size(); i++) {
        if (input.data[i] > 0) {
            output.data[i] = input.data[i];
            mask_.data[i] = 1.0; // Mark as active
        }
        else {
            output.data[i] = 0.0;
            mask_.data[i] = 0.0; // Mark as inactive
        }
    }

    return output;
}

Tensor3D ReLUActivationLayer::backward(const Tensor3D& gradient_output) const {
    Tensor3D gradient_input(gradient_output.channels, gradient_output.height, gradient_output.width);

    for (int i = 0; i < gradient_output.size(); i++) {
        gradient_input.data[i] = mask_.data[i] * gradient_output.data[i];
    }

    return gradient_input;
}