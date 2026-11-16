// Bildet den Orchestrator für die Layers, Activation, Softmax und Loss - Funktionen
#pragma once

#include "matrix/matrix.hpp"
#include "layer/layer.hpp"
#include "activation/relu.hpp"
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

template<class OptimizerType>
class NeuralNetwork{
    public:

        NeuralNetwork(std::vector<std::size_t> layer_sizes,
            size_t number_labels,
            ActivationType activation,
            LossType loss,
            size_t batch_size=64)
        : number_labels(number_labels), batch_size(batch_size), loss_type(loss) {

            if (layer_sizes.size() < 3)
                throw std::invalid_argument("Es muss mindestens 3 Layer geben");

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

            Layers.reserve(layer_sizes.size());
            for (size_t index=1; index < layer_sizes.size(); index++){
                Layers.emplace_back(layer_sizes[index - 1], layer_sizes[index], batch_size, *activation_);
            }

        }

        //trainiert das Neuronale Netz
        void train(const std::vector<std::vector<float>>&, const std::vector<int>&);

        // Bestimmt den Loss auf dem Testdatensatz
        void test(const std::vector<std::vector<float>>&, const std::vector<int>&);


        // Gibt eine Printe Ausgabe für den(zufälligen) Batch
        void forward_test();

    private:

        std::vector<Layer<OptimizerType>> Layers;

        std::unique_ptr<ActivationFunction> activation_;
        std::unique_ptr<LossFunction> loss_;
        const size_t number_labels;
        const size_t batch_size;
        const LossType loss_type;

        bool trained = false;

        // Gibt den Loss vom Batch an
        float forward(const Matrix<Layout::ColumnMajor>& input, const std::vector<int>& labels, Mode);

        // Berechnet die Ableitungen und passt die Gewichte an
        void backward(const Matrix<Layout::ColumnMajor>& input, const std::vector<int>& labels);
};

#include "neural_network.cpp"
