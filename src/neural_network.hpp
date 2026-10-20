// Bildet den Orchestrator für die Layers, Activation, Softmax und Loss - Funktionen
#pragma once

#include "matrix/matrix.hpp"
#include "convolution/convolution.hpp"
#include "pooling/max_pool.hpp"
#include <span>
#include "layer/layer.hpp"
#include "activation/reLu.hpp"
#include "activation/sigmoid.hpp"
#include "activation/activation.hpp"
#include "loss/loss_function.hpp"
#include "loss/cross_entropy.hpp"
#include "loss/mse.hpp"
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <vector>

enum class LossType{
    MeanSquareError,
    CrossEntropy
};

enum class ActivationType{
    ReLU,
    Sigmoid
};

enum class Mode{
    train,
    test_full,
    test
};

class NeuralNetwork{
    public:

        std::vector<Layer> Layers;
        Convolution convolution;

        // layer_sizes[0] ist die abgeflachte Größe nach dem 2x2-Max-Pooling.
        NeuralNetwork(std::vector<std::size_t> layer_sizes,
            size_t number_labels,
            ActivationType activation,
            LossType loss,
            size_t batch_size=64,
            float learning_rate=0.05,
            size_t kernel_size=3,
            size_t number_kernels=8)
        : convolution(kernel_size, number_kernels), number_labels(number_labels), batch_size(batch_size), learning_rate(learning_rate), loss_type(loss) {

            if (learning_rate <= 0.0f)
                throw std::invalid_argument("Learning rate must be greater than zero");

            if (layer_sizes.size() < 3)
                throw std::invalid_argument("Es muss mindestens 3 Layer geben");

            if (batch_size == 0 || number_labels == 0 || kernel_size == 0 || number_kernels == 0)
                throw std::invalid_argument("Batch-, Klassen- und Kernelgrößen müssen positiv sein");
            if (layer_sizes.back() != number_labels)
                throw std::invalid_argument("Ausgabegröße muss der Anzahl Klassen entsprechen");
            for (size_t size : layer_sizes)
                if (size == 0)
                    throw std::invalid_argument("Layergrößen müssen positiv sein");

            Layers.reserve(layer_sizes.size());
            for (size_t index=1; index < layer_sizes.size(); index++){
                Layers.emplace_back(layer_sizes[index - 1], layer_sizes[index], batch_size);
            }

            if (activation == ActivationType::ReLU)
                activation_ = std::make_unique<ReLU>();
            else if (activation == ActivationType::Sigmoid)
                activation_ = std::make_unique<Sigmoid>();
            else
                throw std::invalid_argument("Aktivierungsfunktion existiert nicht!");

            if (loss == LossType::CrossEntropy)
                loss_ = std::make_unique<CrossEntropy>();
            else if (loss == LossType::MeanSquareError)
                loss_ = std::make_unique<MeanSquareError>();
            else
                throw std::invalid_argument("Lossfunction existiert nicht!");

        }

        std::unique_ptr<ActivationFunction> activation_;
        std::unique_ptr<LossFunction> loss_;
        const size_t number_labels;
        const size_t batch_size;
        const float learning_rate;
        const LossType loss_type;

        bool trained = false;

        // MEC-Daten werden einmal vorbereitet; Rohbilder dienen den Kernelgradienten.
        void train(const std::vector<std::vector<float>>& data, const Images& mec_data,
                   const std::vector<int>& labels, size_t max_samples = 8000 * 64);

        // Wertet den Datensatz in Batches aus, einschließlich des letzten Teilbatches.
        void test(const Images& data, const std::vector<int>& labels);

        // labels enthält nur die Labels der ausgewählten Bilder, in Batch-Reihenfolge.
        float forward(const Images& input, std::span<const size_t> batch_indices,
                      const std::vector<int>& labels, Mode mode);

        // input enthält ursprüngliche Bilder als Spalten, in derselben Batch-Reihenfolge.
        void backward(const Matrix<Layout::ColumnMajor>& input, const std::vector<int>& labels);

    private:
        MaxPool2D pooling_;
        Matrix<Layout::ColumnMajor> pooled_output_{0, 0, Initialization::Zero};
        size_t input_size_ = 0;
        bool backward_ready_ = false;

        void validate_images(const Images& input) const;
        void validate_labels(const std::vector<int>& labels, size_t count) const;
};
