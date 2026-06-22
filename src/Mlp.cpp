#include "Mlp.h"
#include <iostream>
#include <cmath>
#include <iomanip>
#include <stdexcept>
#include <algorithm>
#include <numeric>

MLP::MLP(const vector<int> &layer_sizes, shared_ptr<ActivationFunction> activation, double learning_rate, double momentum, double weight_decay, int seed)
    : layer_sizes_(layer_sizes), activation_(activation), learning_rate_(learning_rate), momentum_(momentum), weight_decay_(weight_decay), num_layers_(layer_sizes.size()), rng_(seed) {
        mt19937 init_rng(seed);
        for (int i = 0; i < num_layers_ - 1; i++) {
        int fan_in = layer_sizes_[i];
        int fan_out = layer_sizes_[i + 1];
        double scale = sqrt(2.0 / fan_in);
        weights_.push_back(random(fan_in, fan_out, scale, init_rng));
        biases_.push_back(zeros(1, fan_out));
        weight_velocities_.push_back(zeros(fan_in, fan_out));
        bias_velocities_.push_back(zeros(1, fan_out));
    }
}

vector<Matrix> MLP::forward(const Matrix &input) const {
    vector<Matrix> activations;
    activations.reserve(num_layers_);
    activations.push_back(input);

    int n_weight_layers = num_layers_ - 1;
    Matrix A = input;

    for (int i = 0; i < n_weight_layers; i++) {
        Matrix Z = A.dot(weights_[i]);
        for (int r = 0; r < Z.rows; r++) {
            for (int c = 0; c < Z.cols; c++) {
                Z.at(r, c) += biases_[i].at(0, c);
            }
        }

        if (i == n_weight_layers - 1) {
            A = SoftMax::forward(Z);
        }
        else {
            A = activation_->forward(Z);
        }

        activations.push_back(A);
    }

    return activations;
}

void MLP::backward(const vector<Matrix> &activations, const Matrix &y_true) {
    int n = y_true.rows;
    int n_layer = num_layers_ - 1;

    vector<Matrix> dW(n_layer);
    vector<Matrix> db(n_layer);

    Matrix delta = activations.back() - y_true;

    for (int i = n_layer - 1; i >= 0; i--) {
        dW[i] = activations[i].transpose().dot(delta) / n;
        db[i] = delta.col_mean();

        if (i > 0) {
            Matrix dA = delta.dot(weights_[i].transpose());
            Matrix derivative = dA.hadamard(activation_->derivative(activations[i]));
            delta = dA.hadamard(derivative);
        } else {
            // Gradiente respecto al input (para CNN backward)
            input_gradient_ = delta.dot(weights_[0].transpose());
        }
    }

    for (int i = 0; i < n_layer; i++) {
        for (int k = 0; k < weights_[i].rows * weights_[i].cols; k++) {
            double grad = dW[i].data[k] + weight_decay_ * weights_[i].data[k];
            weight_velocities_[i].data[k] = momentum_ * weight_velocities_[i].data[k] + grad;
            weights_[i].data[k] -= learning_rate_ * weight_velocities_[i].data[k];
        }
        for (int k = 0; k < biases_[i].rows * biases_[i].cols; k++) {
            bias_velocities_[i].data[k] = momentum_ * bias_velocities_[i].data[k] + db[i].data[k];
            biases_[i].data[k] -= learning_rate_ * bias_velocities_[i].data[k];
        }
    }
}

double MLP::mse_loss(const Matrix &y_pred, const Matrix &y_true) {
    Matrix diff = y_pred - y_true;
    double sum = 0.0;

    for (double value : diff.data) {
        sum += value * value;
    }

    return sum / diff.data.size();
}

Matrix MLP::getInputGradient() const {
    return input_gradient_;
}

void MLP::shuffle_data(Matrix &X, Matrix &y) {
    int n = X.rows;
    vector<int> indexes(n);
    iota(indexes.begin(), indexes.end(), 0);
    shuffle(indexes.begin(), indexes.end(), rng_);

    Matrix X_shuffled(n, X.cols);
    Matrix y_shuffled(n, y.cols);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < X.cols; j++) {
            X_shuffled.at(i, j) = X.at(indexes[i], j);
        }
        for (int j = 0; j < y.cols; j++) {
            y_shuffled.at(i, j) = y.at(indexes[i], j);
        }
    }

    X = X_shuffled;
    y = y_shuffled;
}

HistoryTraining MLP::training(const Matrix &X_train, const Matrix &y_train, int epochs, int batch_size, const Matrix *X_val, const Matrix *y_val, bool verbose, int patience) {
    HistoryTraining history_train;
    bool has_value = (X_val != nullptr && y_val != nullptr);

    Matrix X = X_train;
    Matrix y = y_train;

    int n = X.rows;
    int batches = (n + batch_size - 1) / batch_size;

    double best_value_loss = 1e18;
    int epochs_no_improve = 0;

    vector<Matrix> best_weights = weights_;
    vector<Matrix> best_biases = biases_;

    for (int i = 1; i <= epochs; i++) {
        shuffle_data(X, y);
        double epoch_loss = 0.0;

        for (int b = 0; b < batches; b++) {
            int start = b * batch_size;
            int end = min(start + batch_size, n);
            Matrix X_batch = X.slice(start, end);
            Matrix y_batch = y.slice(start, end);

            auto activations = forward(X_batch);
            double batch_loss = mse_loss(activations.back(), y_batch);
            epoch_loss += batch_loss * (end - start);

            backward(activations, y_batch);
        }

        epoch_loss /= n;
        history_train.train_losses.push_back(epoch_loss);

        double value_loss = 0.0;
        if (has_value) {
            Matrix value_prediction = prediction(*X_val);
            value_loss = mse_loss(value_prediction, *y_val);
            history_train.values_losses.push_back(value_loss);

            if (value_loss < best_value_loss) {
                best_value_loss = value_loss;
                epochs_no_improve = 0;
                best_weights = weights_;
                best_biases = biases_;
            }
            else {
                epochs_no_improve++;
            }

            if (epochs_no_improve >= patience) {
                if (verbose) {
                    cout << "Early stopping at epoch " << i << " (best val loss: " << fixed << setprecision(4) << best_value_loss << ")" << endl;
                }

                weights_ = best_weights;
                biases_ = best_biases;
                break;
            }
        }

        if (verbose && (i % max(1, epochs / 10) == 0 || i == 1)) {
            cout << "Epoch " << i << "/" << epochs << " - Train Loss: " << fixed << setprecision(4) << epoch_loss;
            if (has_value) {
                cout << " - Val Loss: " << fixed << setprecision(4) << value_loss;
            }
            cout << endl;
        }
    }
    
    return history_train;
}

Matrix MLP::prediction(const Matrix &X) const {
    return forward(X).back();
}