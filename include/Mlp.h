#pragma once

#include "Matrix.h"
#include "Activation.h"
#include <vector>
#include <map>
#include <string>
#include <random>
#include <memory>
#include <numeric>

using namespace std;

struct HistoryTraining {
    vector<double> train_losses;
    vector<double> values_losses;
};

class MLP {
    public:
        MLP(const vector<int> &layer_sizes, shared_ptr<ActivationFunction> activation, double learning_rate = 0.01, double momentum = 0.9, double weight_decay = 1e-4, int seed = 42);
        vector<Matrix> forward(const Matrix &input) const;
        void backward(const vector<Matrix> &activations, const Matrix &y_true);
        Matrix getInputGradient() const;

        static double mse_loss(const Matrix &y_pred, const Matrix &y_true);

        HistoryTraining training(const Matrix &X_train, const Matrix &y_train, int epochs = 200, int batch_size = 32, const Matrix *X_val = nullptr, const Matrix *y_val = nullptr, bool verbose = false, int patience = 50);
        Matrix prediction(const Matrix &X) const;
    
    private:
        vector<int> layer_sizes_;
        shared_ptr<ActivationFunction> activation_;
        double learning_rate_;
        double momentum_;
        double weight_decay_;
        int num_layers_;
        vector<Matrix> weights_;
        vector<Matrix> biases_;
        vector<Matrix> weight_velocities_;
        vector<Matrix> bias_velocities_;
        mt19937 rng_;
        Matrix input_gradient_;  // gradiente respecto al input para CNN backward

        void shuffle_data(Matrix &X, Matrix &y);
};