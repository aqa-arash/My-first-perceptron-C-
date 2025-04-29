#include <iostream>
#include "Network/Components.h"
#include "DataHandling.h"
#include "LoopLogger.h"


int main(const int argc, char *argv[]) {
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <path_to_config_file>" << std::endl;
        return 1;
    }
    std::string config_path = argv[1];
    int num_epochs = 0;
    int batch_size = 0;
    int hidden_size = 0;
    double learning_rate = 0.0;
    std::string rel_path_train_images;
    std::string rel_path_train_labels;
    std::string rel_path_test_images;
    std::string rel_path_test_labels;
    std::string rel_path_log_file;

    std::cout << "Reading config file..." << std::endl;
    if (!Utils::parseConfigFile(config_path, num_epochs, batch_size, hidden_size, learning_rate,
                                rel_path_train_images, rel_path_train_labels,
                                rel_path_test_images, rel_path_test_labels,
                                rel_path_log_file)) {
        return 1;
    }

    // Load training data
    std::cout << "Loading data..." << std::endl;
    MNISTLoader train_loader;
    train_loader.loadImages(rel_path_train_images);
    train_loader.loadLabels(rel_path_train_labels);
    const Matrix& train_images = train_loader.images();
    const Matrix& train_labels = train_loader.labels();

    // Create the model
    Network network;
    network.addLayer(784, hidden_size, Activation::relu, Activation::reluDerivative);
    network.addLayer(hidden_size, 10, Activation::identity, Activation::identityDerivative);

    LoopLogger logger(num_epochs);

    // Train the model
    std::cout << "Training..." << std::endl;
    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        Precision total_loss = 0;

        for (int i = 0; i < train_images.rows(); i += batch_size) {
            int current_batch_size = std::min(batch_size, static_cast<int>(train_images.rows() - i));
            Matrix batch_images = train_images.middleRows(i, current_batch_size);
            Matrix batch_labels = train_labels.middleRows(i, current_batch_size);

            // Forward pass
            Matrix logits = network.forward(batch_images);
            Matrix predictions = Activation::softmax(logits);

            // Compute loss
            Vector batch_loss = Loss::crossEntropy(predictions, batch_labels);
            total_loss += batch_loss.sum();

            // Backward pass
            Matrix gradOutput = Loss::softmaxCrossEntropyDerivative(logits, batch_labels);
            network.backward(gradOutput);
            network.updateWeights(learning_rate);
        }

        logger.updateProgress(epoch+1, total_loss / train_images.rows());
    }

    logger.waitForCompletion();

    // Load test data
    std::cout << "Testing..." << std::endl;
    MNISTLoader test_loader;
    test_loader.loadImages(rel_path_test_images);
    test_loader.loadLabels(rel_path_test_labels);
    const Matrix& test_images = test_loader.images();
    const Matrix& test_labels = test_loader.labels();
    int numImages = test_loader.imageCount();

    Matrix predictions = Activation::softmax(network.forward(test_images));

    Eigen::VectorXi pred_indices(predictions.rows());
    Eigen::VectorXi label_indices(test_labels.rows());

    for (int i = 0; i < predictions.rows(); ++i) {
        Eigen::MatrixXd::Index maxIndexPred, maxIndexLabel;
        predictions.row(i).maxCoeff(&maxIndexPred);
        test_labels.row(i).maxCoeff(&maxIndexLabel);
        pred_indices(i) = maxIndexPred;
        label_indices(i) = maxIndexLabel;
    }

    int correct = 0;
    int batchNum = 0;
    std::fstream logFile(rel_path_log_file, std::ios::out);
    if (!logFile) {
        std::cerr << "Error: Could not open log file" << std::endl;
        return 1;
    }
    std::cout << "Writing to log file..." << std::endl;
    for (int i = 0; i < numImages; i += batch_size) {
        logFile << "Current batch: " << batchNum << std::endl;
        batchNum++;
        int current_batch_size = std::min(batch_size, static_cast<int>(test_images.rows() - i));
        Eigen::VectorXi batch_predictions = pred_indices.segment(i,current_batch_size);
        Eigen::VectorXi batch_labels = label_indices.segment(i, current_batch_size);
        for (int j = 0; j < current_batch_size; ++j) {
            logFile<<" - image "<<i+j<< ": Prediction=" << batch_predictions(j) << ". Label=" << batch_labels(j) << std::endl;
            if (batch_predictions(j) == batch_labels(j)) {
                correct++;
            }
        }
    }

    std::cout << "Accuracy: " << static_cast<Precision>(correct) / numImages << std::endl;
    std::cout << "Done!" << std::endl;

    return 0;
}