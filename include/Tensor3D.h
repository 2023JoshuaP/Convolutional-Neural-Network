#pragma once

#include "Matrix.h"

using namespace std;

struct Tensor3D {
    int channels, height, width;
    vector<double> data;

    Tensor3D() : channels(0), height(0), width(0) {}
    Tensor3D(int c, int h, int w, double value = 0.0) : channels(c), height(h), width(w), data(c * h * w, value) {}

    double &at(int c, int h, int w) {
        return data[c * height * width + h * width + w];
    }

    double at(int c, int h, int w) const {
        return data[c * height * width + h * width + w];
    }

    int size() const {
        return channels * height * width;
    }

    Matrix flatten() const;
    static Tensor3D reconstructureFlatMatrix(const Matrix &matrix, int channels, int height, int width);

    Tensor3D operator+(const Tensor3D &other) const;
    Tensor3D operator-(const Tensor3D &other) const;
    Tensor3D operator*(double scalar) const;
};

Tensor3D tensorRandom(int channels, int height, int width, double scale, mt19937 &rng);