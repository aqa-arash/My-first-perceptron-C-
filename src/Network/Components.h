#ifndef PERCEPTRON_COMPONENTS_H
#define PERCEPTRON_COMPONENTS_H

#pragma once

#include <Eigen/Dense>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <functional>
#include "Types.hpp"

namespace Activation {
    inline Matrix relu(const Matrix &x) {
        return x.cwiseMax(0);
    }

    inline Matrix reluDerivative(const Matrix &x) {
        return (x.array() > 0).cast<Precision>();
    }

    inline Matrix softmax(const Matrix &x) {
        Matrix expX = x.array().exp();
        return expX.array().colwise() / expX.rowwise().sum().array();
    }

    // Identity activation for output layer
    inline Matrix identity(const Matrix &x) {
        return x;
    }

    inline Matrix identityDerivative(const Matrix &x) {
        return Matrix::Ones(x.rows(), x.cols());
    }
}

namespace Loss {
    inline Matrix crossEntropyDerivative(const Matrix &predictions, const Matrix &targets) {
        return (-targets).cwiseQuotient(predictions);
    }

    inline Vector crossEntropy(const Matrix &predictions, const Matrix &targets) {
        return -(targets.array() * (predictions.array() + 1e-8).log()).rowwise().sum();
    }

    inline Matrix softmaxCrossEntropyDerivative(const Matrix &logits, const Matrix &targets) {
        Matrix softmax_output = Activation::softmax(logits);
        return softmax_output - targets;
    }
}

class Layer {
public:
    Layer(int inputSize, int outputSize) {
        constexpr int seed = 23405559;
        std::srand(seed);
        // Xavier initialization
        weights = Matrix::Random(inputSize, outputSize) * std::sqrt(2.0 / inputSize);
        biases = Vector::Zero(outputSize);
        gradWeights = Matrix::Zero(inputSize, outputSize);
        gradBiases = Vector::Zero(outputSize);
    }

    Matrix forward(const Matrix &input) {
        inputCache = input;
        return (input * weights).rowwise() + biases.transpose();
    }

    Matrix backward(const Matrix &gradOutput) {
        gradWeights = inputCache.transpose() * gradOutput;
        gradBiases = gradOutput.colwise().sum() / (std::max(static_cast<int>(gradOutput.rows() - 1), 1));
        return gradOutput * weights.transpose();
    }

    void updateWeights(Precision learningRate) {
        weights -= learningRate * gradWeights;
        biases -= learningRate * gradBiases;
    }

private:
    Matrix weights, gradWeights;
    Vector biases, gradBiases;
    Matrix inputCache;
};

class Network {
public:
    void addLayer(int inputSize, int outputSize,
                  std::function<Matrix(const Matrix&)> activation,
                  std::function<Matrix(const Matrix&)> activationDerivative) {
        layers.emplace_back(Layer(inputSize, outputSize), activation, activationDerivative);
    }

    Matrix forward(const Matrix &input) {
        Matrix current = input;
        for (auto &layer : layers) {
            layer.preActivation = layer.layer.forward(current);
            layer.postActivation = layer.activation(layer.preActivation);
            current = layer.postActivation;
        }
        return current; // Output is logits (last layer uses identity)
    }

    void backward(const Matrix &gradOutput) {
        Matrix grad = gradOutput;
        for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
            // Apply activation derivative
            Matrix activationGrad = it->activationDerivative(it->preActivation);
            grad = grad.cwiseProduct(activationGrad);
            // Propagate through layer
            grad = it->layer.backward(grad);
        }
    }

    void updateWeights(Precision learningRate) {
        for (auto &layer : layers) {
            layer.layer.updateWeights(learningRate);
        }
    }

private:
    struct LayerEntry {
        Layer layer;
        std::function<Matrix(const Matrix&)> activation;
        std::function<Matrix(const Matrix&)> activationDerivative;
        Matrix preActivation;  // Stores layer output before activation
        Matrix postActivation; // Stores layer output after activation
    };
    std::vector<LayerEntry> layers;
};

#endif //PERCEPTRON_COMPONENTS_H