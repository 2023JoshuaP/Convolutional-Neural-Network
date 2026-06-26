#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <memory>
#include <random>
#include <opencv2/opencv.hpp>

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
        Dataset train_data = DataLoader::loadFromDirectory("data/train", 8);
        Dataset val_data = DataLoader::loadFromDirectory("data/val", 8);
        Dataset test_data = DataLoader::loadFromDirectory("data/test", 8);

        DataLoader::printDatasetInfo(train_data, "Train");
        DataLoader::printDatasetInfo(val_data, "Validation");
        DataLoader::printDatasetInfo(test_data, "Test");

        ConvolutionalNetwork cnn(0.00005, 0.0, 1e-4, 42);

        cnn.add_convolutional_layer(3, 16, 3, 1, 1);
        cnn.add_ReLU_layer();
        cnn.add_pooling_layer(2, 2, PoolingType::Max);

        cnn.add_convolutional_layer(16, 32, 3, 1, 1);
        cnn.add_ReLU_layer();
        cnn.add_pooling_layer(2, 2, PoolingType::Max);

        auto relu_act = make_shared<ReLU>();
        cnn.build({3, 28, 28}, {128}, 8, relu_act);
        cnn.summary();

        HistoryTraining history = cnn.train(
            train_data.images, train_data.labels,
            15,
            32,
            &val_data.images, &val_data.labels,
            true,
            10
        );

        cout << "\nEvaluación" << endl;
        double train_acc = cnn.accuracy(train_data.images, train_data.labels);
        double val_acc = cnn.accuracy(val_data.images, val_data.labels);
        double test_acc = cnn.accuracy(test_data.images, test_data.labels);

        cout << "Train Accuracy: " << fixed << setprecision(2) << train_acc * 100.0 << "%" << endl;
        cout << "Val   Accuracy: " << fixed << setprecision(2) << val_acc * 100.0 << "%" << endl;
        cout << "Test  Accuracy: " << fixed << setprecision(2) << test_acc * 100.0 << "%" << endl;
        
        vector<string> class_names = {
            "Basophil", "Eosinophil", "Erythroblast", "Immature Granulocytes", 
            "Lymphocyte", "Monocyte", "Neutrophil", "Platelet"
        };
        
        mt19937 rng(42);
        uniform_int_distribution<int> dist(0, test_data.size() - 1);
        
        int cols = 4;
        int rows = 3;
        int cell_w = 220;
        int cell_h = 260;
        cv::Mat canvas(rows * cell_h, cols * cell_w, CV_8UC3, cv::Scalar(30, 30, 30));
        
        for(int i = 0; i < 12; i++) {
            int idx = dist(rng);
            Tensor3D img_tensor = test_data.images[idx];
            
            cv::Mat img(img_tensor.height, img_tensor.width, CV_8UC3);
            for(int c=0; c<3; c++) {
                for(int h=0; h<img_tensor.height; h++) {
                    for(int w=0; w<img_tensor.width; w++) {
                        int bgr_c = (c == 0) ? 2 : ((c == 2) ? 0 : 1);
                        img.at<cv::Vec3b>(h, w)[bgr_c] = static_cast<uchar>(img_tensor.at(c, h, w) * 255.0);
                    }
                }
            }
            
            int predicted = cnn.predict(img_tensor);
            
            int true_label = 0;
            for(int j=0; j<8; j++) {
                if(test_data.labels.at(idx, j) > 0.5) true_label = j;
            }
            
            cv::Mat img_display;
            cv::resize(img, img_display, cv::Size(200, 200), 0, 0, cv::INTER_NEAREST);
            
            int r = i / cols;
            int c = i % cols;
            int x = c * cell_w;
            int y = r * cell_h;
            
            img_display.copyTo(canvas(cv::Rect(x + 10, y + 10, 200, 200)));
            
            string true_text = "R: " + class_names[true_label];
            string pred_text = "P: " + class_names[predicted];
            
            cv::Scalar color = (predicted == true_label) ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);
            
            cv::putText(canvas, true_text, cv::Point(x + 10, y + 230), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255), 1);
            cv::putText(canvas, pred_text, cv::Point(x + 10, y + 250), cv::FONT_HERSHEY_SIMPLEX, 0.4, color, 1);
        }
        
        cv::imwrite("resultados_cnn.png", canvas);
        cv::imshow("Resultados CNN (Grid 3x4) - ESC para salir", canvas);
        cv::waitKey(0);
        cv::destroyAllWindows();
    }
    catch (const exception& e) {
        cerr << "\n✗ ERROR: " << e.what() << endl;
        return 1;
    }

    return 0;
}
