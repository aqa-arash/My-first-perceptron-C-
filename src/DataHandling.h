#ifndef PERCEPTRON_DATAHANDLING_H
#define PERCEPTRON_DATAHANDLING_H

#pragma once
#include <string>
#include <fstream>
#include <vector>
#include <stdexcept>
#include <iostream>
#include <sstream>
#include "Network/Types.hpp"

class MNISTLoader {
public:
    MNISTLoader() = default;

    MNISTLoader(const MNISTLoader& other) = delete;
    MNISTLoader& operator=(const MNISTLoader& other) = delete;
    MNISTLoader(MNISTLoader&& other) = delete;
    MNISTLoader& operator=(MNISTLoader&& other) = delete;

    void loadImages(const std::string& path) {
        int magicNumber, count, rows, cols;
        readHeader(path, magicNumber, count, rows, cols, true);
        images_ = loadMnistImages(path, count, rows, cols);
        image_count_ = count;
        rows_ = rows;
        cols_ = cols;
    }

    void loadLabels(const std::string& path) {
        int magicNumber, count, rows, cols;
        readHeader(path, magicNumber, count, rows, cols, false);
        labels_ = loadMnistLabels(path, count);
        label_count_ = count;
    }

    [[nodiscard]] const Matrix& images() const { return images_; }
    [[nodiscard]] const Matrix& labels() const { return labels_; }
    [[nodiscard]] int imageCount() const { return image_count_; }
    [[nodiscard]] int labelCount() const { return label_count_; }
    [[nodiscard]] int rows() const { return rows_; }
    [[nodiscard]] int cols() const { return cols_; }

private:
    Matrix images_;
    Matrix labels_;
    int image_count_ = 0;
    int label_count_ = 0;
    int rows_ = 0, cols_ = 0;

    static void readHeader(const std::string &path, int &magicNumber, int &numItems, int &rows, int &cols, bool isImages) {
        std::ifstream file(path, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open file: " + path);
        }
        file.read(reinterpret_cast<char *>(&magicNumber), 4);
        file.read(reinterpret_cast<char *>(&numItems), 4);
        magicNumber = static_cast<int>(__builtin_bswap32(magicNumber));
        numItems = static_cast<int>(__builtin_bswap32(numItems));
        if (isImages) {
            file.read(reinterpret_cast<char *>(&rows), 4);
            file.read(reinterpret_cast<char *>(&cols), 4);
            rows = static_cast<int>(__builtin_bswap32(rows));
            cols = static_cast<int>(__builtin_bswap32(cols));
        } else {
            rows = cols = 0;
        }
        file.close();
    }

    static Matrix loadMnistImages(const std::string &filePath, const int numImages, const int rows, const int cols) {
        std::ifstream file(filePath, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open file: " + filePath);
        }

        file.seekg(16, std::ios::beg);

        Matrix images(numImages, rows * cols);
        for (int i = 0; i < numImages; ++i) {
            std::vector<unsigned char> buffer(rows * cols);
            file.read(reinterpret_cast<char *>(buffer.data()), rows * cols);
            for (int j = 0; j < rows * cols; ++j) {
                images(i, j) = static_cast<Precision>(buffer[j] / 255.0);
            }
        }

        file.close();
        return images;
    }

    static Matrix loadMnistLabels(const std::string &filePath, const int numLabels) {
        std::ifstream file(filePath, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open file: " + filePath);
        }

        file.seekg(8, std::ios::beg);

        Matrix labels = Matrix::Zero(numLabels, 10);
        for (int i = 0; i < numLabels; ++i) {
            unsigned char label;
            file.read(reinterpret_cast<char *>(&label), 1);
            int j = static_cast<int>(label);
            labels(i, j) = 1;
        }

        file.close();
        return labels;
    }
};

namespace Utils {
    inline void writeTensorToFile(const Matrix& tensor, const std::string& filename) {
        std::ofstream file(filename);
        if(!file) {
            throw std::runtime_error("Failed to create output file: " + filename);
        }

        int rows = tensor.rows();
        int cols = tensor.cols();

        if (rows==1 && cols ==10 ) {
            file << "1\n10"<<std::endl;
        }
        else {
            file << "2\n"<<rows<<"\n"<<cols<<std::endl;
        }

        for (int col = 0 ; col < cols; ++col){
            for (int row = 0; row < rows ; ++row) {
                file << tensor(row, col) << std::endl;
            }
        }
        file.close();
    }

    inline bool parseConfigFile(const std::string& filename,
                                int & num_epochs,
                                int & batch_size,
                                int & hidden_size ,
                                double & learning_rate,
                                std::string & rel_path_train_images,
                                std::string & rel_path_train_labels ,
                                std::string & rel_path_test_images,
                                std::string & rel_path_test_labels,
                                std::string & rel_path_log_file ) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error opening file: " << filename << std::endl;
            return false;
        }

        std::string line;
        while (std::getline(file, line)) {
            if (line.empty() || line[0] == '#') {
                continue;
            }

            std::istringstream iss(line);
            std::string key, value;

            if (std::getline(iss, key, '=') && std::getline(iss, value)) {
                key.erase(0, key.find_first_not_of(" \t"));
                key.erase(key.find_last_not_of(" \t") + 1);
                value.erase(0, value.find_first_not_of(" \t"));
                value.erase(value.find_last_not_of(" \t") + 1);

                if (key == "num_epochs") {
                    num_epochs = std::stoi(value);
                } else if (key == "batch_size") {
                    batch_size = std::stoi(value);
                } else if (key == "hidden_size") {
                    hidden_size = std::stoi(value);
                } else if (key == "learning_rate") {
                    learning_rate = std::stod(value);
                } else if (key == "rel_path_train_images") {
                    rel_path_train_images = value;
                } else if (key == "rel_path_train_labels") {
                    rel_path_train_labels = value;
                } else if (key == "rel_path_test_images") {
                    rel_path_test_images = value;
                } else if (key == "rel_path_test_labels") {
                    rel_path_test_labels = value;
                } else if (key == "rel_path_log_file") {
                    rel_path_log_file = value;
                } else {
                    std::cerr << "Unknown key: " << key << std::endl;
                }
            }
        }

        file.close();
        return true;
    }
} // namespace Utils

#endif //PERCEPTRON_DATAHANDLING_H
