#pragma once

#include <vector>
#include <stdexcept>
#include <cmath>
#include <random>
#include <functional>

using namespace std;

struct Matrix {
    int rows, cols;
    vector<double> data;

    Matrix() : rows(0), cols(0) {}
    Matrix(int r, int c, double value = 0.0) : rows(r), cols(c), data(r * c, value) {}

    double &at(int r, int c) {
        if (r < 0 || r >= rows || c < 0 || c >= cols) {
            throw out_of_range("Index out of bounds");
        }
        return data[r * cols + c];
    }

    double at(int r, int c) const {
        if (r < 0 || r >= rows || c < 0 || c >= cols) {
            throw out_of_range("Index out of bounds");
        }
        return data[r * cols + c];
    }

    Matrix operator+(const Matrix &other) const;
    Matrix operator-(const Matrix &other) const;
    Matrix operator*(double scalar) const;
    Matrix operator/(double scalar) const;

    Matrix hadamard(const Matrix &other) const;
    Matrix dot(const Matrix &other) const;
    Matrix transpose() const;
    Matrix applyFunction(const function<double(double)> &func) const;
    Matrix col_mean() const;

    Matrix slice(int row_start, int row_end) const;

    double sum() const;
    double mean() const;
};

Matrix zeros(int rows, int cols);
Matrix random(int rows, int cols, double scale, mt19937 &rng);