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

Tensor3D Tensor3D::operator+(const Tensor3D &other) const {
    Tensor3D result(channels, height, width);
    for (int i = 0; i < size(); i++) {
        result.data[i] = data[i] + other.data[i];
    }

    return result;
}

Tensor3D Tensor3D::operator-(const Tensor3D &other) const {
    Tensor3D result(channels, height, width);
    for (int i = 0; i < size(); i++) {
        result.data[i] = data[i] - other.data[i];
    }

    return result;
}

Tensor3D Tensor3D::operator*(double scalar) const {
    Tensor3D result(channels, height, width);
    for (int i = 0; i < size(); i++) {
        result.data[i] = data[i] * scalar;
    }

    return result;
}

Tensor3D tensorRandom(int channels, int height, int width, double scale, mt19937 &rng) {
    Tensor3D result(channels, height, width);
    normal_distribution<double> distance(0.0, scale);
    for (int i = 0; i < result.size(); i++) {
        result.data[i] = distance(rng);
    }

    return result;
}