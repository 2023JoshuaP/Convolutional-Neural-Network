#include "Tensor3D.h"

Matrix Tensor3D::flatten() const {
    Matrix result(1, size());
    for (int i = 0; i < size(); ++i) {
        result.data[i] = data[i];
    }

    return result;
}

Tensor3D Tensor3D::reconstructureFlatMatrix(const Matrix &matrix, int channels, int height, int width) {
    Tensor3D result(channels, height, width);
    int total = channels * height * width;
    for (int i = 0; i < total; ++i) {
        result.data[i] = matrix.data[i];
    }
    
    return result;
}