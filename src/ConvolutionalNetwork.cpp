#include "ConvolutionalNetwork.h"

ConvolutionalNetwork::ConvolutionalNetwork(double learning_rate, double momentum, double weight_decay, int seed)
    : learning_rate_(learning_rate), momentum_(momentum), weight_decay_(weight_decay), seed_(seed), built_(false), flatten_size_(0) {}

void ConvolutionalNetwork::add_convolutional_layer(int in_channels, int out_channels, int kernel_size, int stride, int padding) {
    int index = conv_layers_.size();
    conv_layers_.emplace_back(in_channels, out_channels, kernel_size, stride, padding, seed_ + index);
    layer_order_.emplace_back(LayerType::Convolution, index);
}

void ConvolutionalNetwork::add_ReLU_layer() {
    int index = activation_layers_.size();
    activation_layers_.emplace_back();
    layer_order_.emplace_back(LayerType::ReLU, index);
}

void ConvolutionalNetwork::add_pooling_layer(int pool_size, int stride, PoolingType type) {
    int index = pooling_layers_.size();
    pooling_layers_.emplace_back(pool_size, stride, type);
    layer_order_.emplace_back(LayerType::Pooling, index);
}

void ConvolutionalNetwork::build(const vector<int> &input_shape, const vector<int> &mlp_hidden_sizes, int num_classes, shared_ptr<ActivationFunction> mlp_activation) {
    if (input_shape.size() != 3) {
        throw invalid_argument("Input shape must be a vector of 3 integers: [channels, height, width]");
    }

    Tensor3D dummy_input(input_shape[0], input_shape[1], input_shape[2]);

    for (auto &[type, index] : layer_order_) {
        switch (type) {
            case LayerType::Convolution:
                dummy_input = conv_layers_[index].forward(dummy_input);
                break;
            case LayerType::ReLU:
                dummy_input = activation_layers_[index].forward(dummy_input);
                break;
            case LayerType::Pooling:
                dummy_input = pooling_layers_[index].forward(dummy_input);
                break;
        }
    }

    flatten_size_ = dummy_input.channels * dummy_input.height * dummy_input.width;
    vector<int> mlp_sizes;
    mlp_sizes.push_back(flatten_size_);

    for (int h : mlp_hidden_sizes) {
        mlp_sizes.push_back(h);
    }
    mlp_sizes.push_back(num_classes);

    mlp_classificator_ = make_unique<MLP>(mlp_sizes, mlp_activation,learning_rate_, momentum_, weight_decay_, seed_);
    built_ = true;

    cout << "Convolutiona Network built. Flatten size: " << flatten_size_ << " (" << dummy_input.channels << " x " << dummy_input.height << " x " << dummy_input.width << ")" << endl;
}

Matrix ConvolutionalNetwork::forward(const Tensor3D &input) {
    if (!built_) {
        throw runtime_error("Network must be built before calling forward.");
    }

    layer_inputs_.clear();
    Tensor3D current_input = input;

    for (auto &[type, index] : layer_order_) {
        layer_inputs_.push_back(current_input);

        switch (type) {
            case LayerType::Convolution:
                current_input = conv_layers_[index].forward(current_input);
                break;
            case LayerType::ReLU:
                current_input = activation_layers_[index].forward(current_input);
                break;
            case LayerType::Pooling:
                current_input = pooling_layers_[index].forward(current_input);
                break;
        }
    }

    flatten_input_cache_ = Tensor3D(current_input.channels, current_input.height, current_input.width);
    Matrix flat = current_input.flatten();
    mlp_activations_ = mlp_classificator_->forward(flat);
    return mlp_activations_.back();
}

void ConvolutionalNetwork::backward(const Matrix &y_true) {
    if (!built_) {
        throw runtime_error("Network must be built before calling backward.");
    }

    mlp_classificator_ -> backward(mlp_activations_, y_true);
    Matrix grad_flat = mlp_classificator_->getInputGradient();

    Tensor3D grad = Tensor3D::reconstructureFlatMatrix(
        grad_flat, flatten_input_cache_.channels, flatten_input_cache_.height, flatten_input_cache_.width
    );

    for (int i = layer_order_.size() - 1; i >= 0; i--) {
        auto &[type, index] = layer_order_[i];
        switch (type) {
            case LayerType::Convolution:
                grad = conv_layers_[index].backward(grad, learning_rate_);
                break;
            case LayerType::ReLU:
                grad = activation_layers_[index].backward(grad);
                break;
            case LayerType::Pooling:
                grad = pooling_layers_[index].backward(grad);
                break;
        }
    }
}

HistoryTraining ConvolutionalNetwork::train(const vector<Tensor3D> &X_train, const Matrix &y_train, int epochs, int batch_size,
    const vector<Tensor3D> *X_value, const Matrix *y_value, bool verbose, int patience) {
    if (!built_) {
        throw runtime_error("Network must be built before calling train.");
    }

    HistoryTraining history;
    int n = X_train.size();
    bool has_val = (X_value != nullptr && y_value != nullptr);

    double best_val_loss = 1e18;
    int epochs_no_improve = 0;

    vector<int> indices(n);
    iota(indices.begin(), indices.end(), 0);
    mt19937 rng(seed_);

    for (int epoch = 1; epoch <= epochs; epoch++) {
        shuffle(indices.begin(), indices.end(), rng);
        double epoch_loss = 0.0;

        int batches = (n + batch_size - 1) / batch_size;

        for (int b = 0; b < batches; b++) {
            int start = b * batch_size;
            int end = min(start + batch_size, n);
            int current_batch_size = end - start;

            double batch_loss = 0.0;

            for (int i = start; i < end; i++) {
                int index = indices[i];
                Matrix output = forward(X_train[index]);
                Matrix y_sample(1, y_train.cols);

                for (int j = 0; j < y_train.cols; j++) {
                    y_sample.at(0, j) = y_train.at(index, j);
                }

                batch_loss += MLP::mse_loss(output, y_sample);
                backward(y_sample);
            }

            epoch_loss += batch_loss;
        }

        epoch_loss /= n;
        history.train_losses.push_back(epoch_loss);

        double val_loss = 0.0;
        if (has_val) {
            int n_val = X_value->size();
            for (int i = 0; i < n_val; i++) {
                Matrix output = forward((*X_value)[i]);
                Matrix y_sample(1, y_value->cols);

                for (int j = 0; j < y_value->cols; j++) {
                    y_sample.at(0, j) = (*y_value).at(i, j);
                }

                val_loss += MLP::mse_loss(output, y_sample);
            }

            val_loss /= n_val;
            history.values_losses.push_back(val_loss);
            if (val_loss < best_val_loss) {
                best_val_loss = val_loss;
                epochs_no_improve = 0;
            }
            else {
                epochs_no_improve++;
            }

            if (epochs_no_improve >= patience) {
                if (verbose) {
                    cout << "Early stopping at epoch " << epoch << " with validation loss: " << fixed << setprecision(4) << best_val_loss << endl;
                }
                break;
            }
        }

        if (verbose && (epoch % max(1, epochs / 10) == 0) || epoch == 1) {
            cout << "Epoch " << epoch << "/" << epochs << " - Train Loss: " << fixed << setprecision(4) << epoch_loss;
            if (has_val) {
                cout << " - Val Loss: " << fixed << setprecision(4) << val_loss;
            }
            cout << endl;
        }
    }

    return history;
}

int ConvolutionalNetwork::predict(const Tensor3D &input) {
    Matrix output = forward(input);
    int best_class = 0;
    double best_score = output.at(0, 0);

    for (int j = 1; j < output.cols; j++) {
        if (output.at(0, j) > best_score) {
            best_score = output.at(0, j);
            best_class = j;
        }
    }

    return best_class;
}

double ConvolutionalNetwork::accuracy(const vector<Tensor3D> &X, const Matrix &y) {
    int correct = 0;
    int total = X.size();

    for (size_t i = 0; i < X.size(); i++) {
        int pred_class = predict(X[i]);
        int true_class = 0;
        double best_score = y.at(i, 0);

        for (int j = 1; j < y.cols; j++) {
            if (y.at(i, j) > best_score) {
                best_score = y.at(i, j);
                true_class = j;
            }
        }

        if (pred_class == true_class) {
            correct++;
        }
    }

    return static_cast<double>(correct) / total;
}

void ConvolutionalNetwork::summary() const {
    cout << "Convolutional Neural Network Summary:" << endl;
    int layer_num = 1;
    for (auto &[type, index] : layer_order_) {
        switch (type) {
            case LayerType::Convolution: {
                cout << "Layer " << layer_num++ << ": Conv2D(k=)" << conv_layers_[index].get_kernel() << ", s=" << conv_layers_[index].get_stride() << ", p=" << conv_layers_[index].get_padding()
                    << ", out_channels=" << conv_layers_[index].get_output_channels() << endl;
                    break;
            }
            case LayerType::ReLU:
                cout << "Layer " << layer_num++ << ": ReLU Activation" << endl;
                break;
            case LayerType::Pooling: {
                cout << "Layer " << layer_num++ << ": Pooling Layer " << pooling_layers_[index].getTypeName() << endl;
                break;
            }
        }
    }

    cout << "Flatten size: " << flatten_size_ << endl;
    cout << "MLP Classificator" << endl;
}