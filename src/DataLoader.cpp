#include "DataLoader.h"
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <algorithm>
#include <iostream>
#include <random>

namespace fs = std::filesystem;

Dataset DataLoader::loadFromDirectory(const string& base_path, int num_classes) {
    Dataset dataset;
    dataset.num_classes = num_classes;

    vector<pair<string, int>> image_paths;

    for (int cls = 0; cls < num_classes; cls++) {
        string class_dir = base_path + "/" + to_string(cls);

        if (!fs::exists(class_dir)) {
            cerr << "Advertencia: directorio de clase no encontrado: " << class_dir << endl;
            continue;
        }

        for (const auto& entry : fs::directory_iterator(class_dir)) {
            if (entry.is_regular_file()) {
                string ext = entry.path().extension().string();
                if (ext == ".png" || ext == ".jpg" || ext == ".jpeg" || ext == ".bmp") {
                    image_paths.push_back({entry.path().string(), cls});
                }
            }
        }
    }

    sort(image_paths.begin(), image_paths.end());

    int n = image_paths.size();
    if (n == 0) {
        throw runtime_error("No se encontraron imágenes en: " + base_path);
    }

    cv::Mat sample = cv::imread(image_paths[0].first, cv::IMREAD_COLOR);
    if (sample.empty()) {
        throw runtime_error("No se pudo leer la imagen: " + image_paths[0].first);
    }

    dataset.height = sample.rows;
    dataset.width = sample.cols;
    dataset.channels = sample.channels();

    dataset.labels = Matrix(n, num_classes, 0.0);

    dataset.images.reserve(n);

    for (int i = 0; i < n; i++) {
        const auto& [path, label] = image_paths[i];

        cv::Mat img = cv::imread(path, cv::IMREAD_COLOR);
        if (img.empty()) {
            cerr << "Advertencia: no se pudo leer " << path << ", saltando." << endl;
            continue;
        }

        cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

        Tensor3D tensor(dataset.channels, dataset.height, dataset.width);

        for (int c = 0; c < dataset.channels; c++) {
            for (int h = 0; h < dataset.height; h++) {
                for (int w = 0; w < dataset.width; w++) {
                    tensor.at(c, h, w) = img.at<cv::Vec3b>(h, w)[c] / 255.0;
                }
            }
        }

        dataset.images.push_back(tensor);

        int idx = dataset.images.size() - 1;
        dataset.labels.at(idx, label) = 1.0;

        if ((i + 1) % 2000 == 0 || i == n - 1) {
            cout << "  Cargando: " << (i + 1) << "/" << n << " imágenes\r" << flush;
        }
    }

    cout << endl;

    if ((int)dataset.images.size() < n) {
        Matrix adjusted_labels(dataset.images.size(), num_classes, 0.0);
        for (int i = 0; i < (int)dataset.images.size(); i++) {
            for (int j = 0; j < num_classes; j++) {
                adjusted_labels.at(i, j) = dataset.labels.at(i, j);
            }
        }
        dataset.labels = adjusted_labels;
    }

    return dataset;
}

void DataLoader::printDatasetInfo(const Dataset& dataset, const string& name) {
    cout << "Dataset: " << name << endl;
    cout << "  Muestras: " << dataset.size() << endl;
    cout << "  Dimensiones: " << dataset.channels << "x" << dataset.height << "x" << dataset.width << endl;
    cout << "  Clases: " << dataset.num_classes << endl;

    vector<int> class_count(dataset.num_classes, 0);
    for (int i = 0; i < dataset.size(); i++) {
        for (int j = 0; j < dataset.num_classes; j++) {
            if (dataset.labels.at(i, j) > 0.5) {
                class_count[j]++;
                break;
            }
        }
    }
    cout << "  Distribución por clase: ";
    for (int j = 0; j < dataset.num_classes; j++) {
        cout << j << ":" << class_count[j] << " ";
    }
    cout << endl;
}