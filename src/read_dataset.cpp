#include <iostream>
#include <filesystem>
#include "DataHandling.h"

int main(const int argc, char *argv[]) {
    if (argc == 5) {
        const std::string inputPath = argv[1];
        const std::string outputPath = argv[2];
        const int index = std::stoi(argv[3]);
        const bool isImage = (atoi(argv[4]) != 0);

        try {
            MNISTLoader loader;
            if (isImage) {
                loader.loadImages(inputPath);
                const auto dataset = loader.images();
                Utils::writeTensorToFile(dataset.row(index).reshaped(loader.rows(),loader.cols()),outputPath);

            } else {
                loader.loadLabels(inputPath);
                int rows = 1;
                int cols=10;
                const auto dataset = loader.labels();
                Utils::writeTensorToFile(dataset.row(index).reshaped(rows,cols),outputPath);
            }

        } catch (const std::exception &e) {
            std::cerr << "Error: " << e.what() << "\n";
        }
        return 0;
    } else {
        std::cout << "incorrect number of arguments" << std::endl;
        return -1;
    }
}
