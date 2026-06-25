#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <memory>
#include <chrono>

#include "Tensor3D.h"
#include "ConvolutionalLayer.h"
#include "ActivationLayer.h"
#include "PoolingLayer.h"
#include "ConvolutionalNetwork.h"
#include "Activation.h"
#include "DataLoader.h"

using namespace std;

int main() {
    try {
        // ─── Cargar BloodMNIST con OpenCV ───
        cout << "=== Cargando BloodMNIST ===" << endl;

        auto t_start = chrono::high_resolution_clock::now();

        Dataset train_data = DataLoader::loadFromDirectory("data/train", 8);
        Dataset val_data = DataLoader::loadFromDirectory("data/val", 8);
        Dataset test_data = DataLoader::loadFromDirectory("data/test", 8);

        auto t_load = chrono::high_resolution_clock::now();
        double load_time = chrono::duration<double>(t_load - t_start).count();
        cout << "Tiempo de carga: " << fixed << setprecision(2) << load_time << "s\n" << endl;

        DataLoader::printDatasetInfo(train_data, "Train");
        DataLoader::printDatasetInfo(val_data, "Validation");
        DataLoader::printDatasetInfo(test_data, "Test");

        // ─── Definir arquitectura CNN ───
        cout << "\n=== Construyendo CNN ===" << endl;

        ConvolutionalNetwork cnn(0.001, 0.9, 1e-4, 42);

        // Bloque 1: Conv(3→16, k=3, p=1) → ReLU → MaxPool(2,2)
        // Input: 3x28x28 → Conv → 16x28x28 → Pool → 16x14x14
        cnn.add_convolutional_layer(3, 16, 3, 1, 1);
        cnn.add_ReLU_layer();
        cnn.add_pooling_layer(2, 2, PoolingType::Max);

        // Bloque 2: Conv(16→32, k=3, p=1) → ReLU → MaxPool(2,2)
        // 16x14x14 → Conv → 32x14x14 → Pool → 32x7x7
        cnn.add_convolutional_layer(16, 32, 3, 1, 1);
        cnn.add_ReLU_layer();
        cnn.add_pooling_layer(2, 2, PoolingType::Max);

        // Build: flatten(32x7x7=1568) → MLP(1568 → 128 → 8)
        auto relu_act = make_shared<ReLU>();
        cnn.build({3, 28, 28}, {128}, 8, relu_act);
        cnn.summary();

        // ─── Entrenar ───
        cout << "\n=== Entrenamiento ===" << endl;
        auto t_train_start = chrono::high_resolution_clock::now();

        HistoryTraining history = cnn.train(
            train_data.images, train_data.labels,
            20,         // epochs
            32,         // batch_size
            &val_data.images, &val_data.labels,
            true,       // verbose
            10          // patience (early stopping)
        );

        auto t_train_end = chrono::high_resolution_clock::now();
        double train_time = chrono::duration<double>(t_train_end - t_train_start).count();
        cout << "\nTiempo de entrenamiento: " << fixed << setprecision(2) << train_time << "s" << endl;

        // ─── Evaluar ───
        cout << "\n=== Evaluación ===" << endl;
        double train_acc = cnn.accuracy(train_data.images, train_data.labels);
        double val_acc = cnn.accuracy(val_data.images, val_data.labels);
        double test_acc = cnn.accuracy(test_data.images, test_data.labels);

        cout << "Train Accuracy: " << fixed << setprecision(2) << train_acc * 100.0 << "%" << endl;
        cout << "Val   Accuracy: " << fixed << setprecision(2) << val_acc * 100.0 << "%" << endl;
        cout << "Test  Accuracy: " << fixed << setprecision(2) << test_acc * 100.0 << "%" << endl;

        cout << "\n=============================" << endl;
        cout << "  Ejecución completada ✓" << endl;
        cout << "=============================" << endl;
    }
    catch (const exception& e) {
        cerr << "\n✗ ERROR: " << e.what() << endl;
        return 1;
    }

    return 0;
}
