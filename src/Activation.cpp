#include "Activation.h"

Matrix Sigmoid::forward(const Matrix &z) const {
    return z.applyFunction([](double x) {
        return 1.0 / (1.0 + exp(-x));
    });
}

Matrix Sigmoid::derivative(const Matrix &a) const {
    return a.applyFunction([](double x) {
        return x * (1.0 - x);
    });
}

Matrix ReLU::forward(const Matrix &z) const {
    return z.applyFunction([](double x) {
        return max(0.0, x);
    });
}

Matrix ReLU::derivative(const Matrix &a) const {
    return a.applyFunction([](double x) {
        return x > 0.0 ? 1.0 : 0.0;
    });
}

Matrix SoftMax::forward(const Matrix &z) {
    Matrix result(z.rows, z.cols);
    for (int i = 0; i < z.rows; i++) {
        double max_val = z.at(i, 0);
        for (int j = 1; j < z.cols; j++) {
            max_val = max(max_val, z.at(i, j));
        }

        double sum_exp = 0.0;
        for (int j = 0; j < z.cols; j++) {
            result.at(i, j) = exp(z.at(i, j) - max_val);
            sum_exp += result.at(i, j);
        }

        for (int j = 0; j < z.cols; j++) {
            result.at(i, j) /= sum_exp;
        }
    }

    return result;
}