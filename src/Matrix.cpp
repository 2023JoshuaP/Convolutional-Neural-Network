#include "Matrix.h"

Matrix Matrix::operator+(const Matrix &other) const {
    if (rows != other.rows || cols != other.cols) {
        throw invalid_argument("Matrix dimensions must match for addition");
    }

    Matrix result(rows, cols);
    for (int i = 0; i < rows * cols; ++i) {
        result.data[i] = data[i] + other.data[i];
    }

    return result;
}

Matrix Matrix::operator-(const Matrix &other) const {
    if (rows != other.rows || cols != other.cols) {
        throw invalid_argument("Matrix dimensions must match for subtraction");
    }

    Matrix result(rows, cols);
    for (int i = 0; i < rows * cols; ++i) {
        result.data[i] = data[i] - other.data[i];
    }

    return result;
}

Matrix Matrix::operator*(double scalar) const {
    Matrix result(rows, cols);
    for (int i = 0; i < rows * cols; ++i) {
        result.data[i] = data[i] * scalar;
    }
    return result;
}

Matrix Matrix::operator/(double scalar) const {
    if (scalar == 0) {
        throw invalid_argument("Division by zero");
    }

    Matrix result(rows, cols);
    for (int i = 0; i < rows * cols; ++i) {
        result.data[i] = data[i] / scalar;
    }
    return result;
}

Matrix Matrix::hadamard(const Matrix &other) const {
    if (rows != other.rows || cols != other.cols) {
        throw invalid_argument("Matrix dimensions must match for Hadamard product");
    }

    Matrix result(rows, cols);
    for (int i = 0; i < rows * cols; ++i) {
        result.data[i] = data[i] * other.data[i];
    }

    return result;
}

Matrix Matrix::dot(const Matrix &other) const {
    if (cols != other.rows) {
        throw invalid_argument("Inner matrix dimensions must match for dot product");
    }

    Matrix result(rows, other.cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < other.cols; j++) {
            double sum = 0.0;
            for (int k = 0; k < cols; k++) {
                sum += at(i, k) * other.at(k, j);
            }
            result.at(i, j) = sum;
        }
    }

    return result;
}

Matrix Matrix::transpose() const {
    Matrix result(cols, rows);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result.at(j, i) = at(i, j);
        }
    }

    return result;
}

Matrix Matrix::applyFunction(const function<double(double)> &func) const {
    Matrix result(rows, cols);
    for (int i = 0; i < rows * cols; ++i) {
        result.data[i] = func(data[i]);
    }

    return result;
}

Matrix Matrix::col_mean() const {
    Matrix result(1, cols, 0.0);
    for (int j = 0; j < cols; j++) {
        double sum = 0.0;
        for (int i = 0; i < rows; i++) {
            sum += at(i, j);
        }
        result.at(0, j) = sum / rows;
    }

    return result;
}

Matrix Matrix::slice(int row_start, int row_end) const {
    int n = row_end - row_start;
    Matrix result(n, cols);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < cols; j++) {
            result.at(i, j) = at(row_start + i, j);
        }
    }

    return result;
}

double Matrix::sum() const {
    double total = 0.0;
    for (const auto &value : data) {
        total += value;
    }

    return total;
}

double Matrix::mean() const {
    if (rows == 0 || cols == 0) {
        throw invalid_argument("Cannot compute mean of an empty matrix");
    }
    return sum() / (rows * cols);
}

Matrix zeros(int rows, int cols) {
    return Matrix(rows, cols, 0.0);
}

Matrix random(int rows, int cols, double scale, mt19937 &rng) {
    Matrix result(rows, cols);
    uniform_real_distribution<double> dist(-scale, scale);
    for (int i = 0; i < rows * cols; ++i) {
        result.data[i] = dist(rng);
    }
    
    return result;
}