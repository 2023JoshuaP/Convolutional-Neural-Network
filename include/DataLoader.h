#pragma once

#include "Tensor3D.h"
#include <vector>
#include <string>

using namespace std;

struct Dataset {
    vector<Tensor3D> images;
    Matrix labels;
    int num_classes;
    int channels, height, width;
    int size() const { return images.size(); }
};

class DataLoader {
public:
    static Dataset loadFromDirectory(const string& base_path, int num_classes);
    static void printDatasetInfo(const Dataset& dataset, const string& name);
};