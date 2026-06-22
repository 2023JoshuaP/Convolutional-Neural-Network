#pragma once

#include "Tensor3D.h"
#include <algorithm>

class ReLUActivationLayer {
    public:
        Tensor3D forward(const Tensor3D& input);
        Tensor3D backward(const Tensor3D& gradient_output) const;
    
    private:
        Tensor3D mask_;
};