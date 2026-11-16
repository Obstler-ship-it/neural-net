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
#include <type_traits>
#include <vector>

struct Dense {};

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

struct test_result{
    float precision;
    float time;
    size_t size;
};

template<class OptimizerType>
class NeuralNetwork{
    public:

        // @Param input_size, batch_size, epoch_without_improvement
        NeuralNetwork(size_t input_size, size_t batch_size=64, int epochs_without_improvement=2,
                      const typename OptimizerType::Config& optimizer_config = {})
        : number_labels(0), batch_size(batch_size),
          epochs_without_improvement(epochs_without_improvement),
          loss_type(LossType::CrossEntropy),
          input_size(input_size), current_size(input_size), optimizer_config(optimizer_config),
          loss_(std::make_unique<CrossEntropy>()) {
            if (!(1 <= epochs_without_improvement))
                throw std::invalid_argument("Epochen-Anzahl ohne Verbesserung muss <= 1 sein");
            if (input_size == 0 || batch_size == 0 )
                throw std::invalid_argument("Eingabe- und Batchgroesse muessen positiv sein");
        }

        template<class Component>
        void add(size_t output_size) {
            static_assert(std::is_same_v<Component, Dense>, "Nur Dense hat eine Groessenangabe");
            if (output_size == 0)
                throw std::invalid_argument("Layergroesse muss positiv sein");
            ReLU default_activation;
            Layers.emplace_back(current_size, output_size, batch_size, default_activation, optimizer_config);
            activations.push_back(nullptr);
            current_size = output_size;
            number_labels = output_size;
        }

        template<class Component>
        void add() {
            static_assert(std::is_same_v<Component, ReLU> || std::is_same_v<Component, Sigmoid>,
                          "Aktivierungsfunktion muss ReLU oder Sigmoid sein");
            if (Layers.empty() || activations.back())
                throw std::logic_error("Aktivierung benoetigt einen Dense-Layer ohne Aktivierung");
            auto activation = std::make_unique<Component>();
            Layers.back().rescale_for(*activation);
            activations.back() = std::move(activation);
        }

        template<class Loss>
        void set_loss() {
            static_assert(std::is_same_v<Loss, CrossEntropy> || std::is_same_v<Loss, MeanSquareError>,
                          "Loss muss CrossEntropy oder MeanSquareError sein");
            loss_ = std::make_unique<Loss>();
            if constexpr (std::is_same_v<Loss, CrossEntropy>)
                loss_type = LossType::CrossEntropy;
            else
                loss_type = LossType::MeanSquareError;
        }

        NeuralNetwork(std::vector<std::size_t> layer_sizes,
            size_t number_labels,
            ActivationType activation,
            LossType loss,
            size_t batch_size=64,
            const typename OptimizerType::Config& optimizer_config = {})
        : number_labels(number_labels), batch_size(batch_size), loss_type(loss),
          input_size(layer_sizes.empty() ? 0 : layer_sizes.front()),
          current_size(layer_sizes.empty() ? 0 : layer_sizes.front()),
          optimizer_config(optimizer_config) {

            if (layer_sizes.size() < 3)
                throw std::invalid_argument("Es muss mindestens 3 Layer geben");
            if (batch_size == 0 || layer_sizes.front() == 0 ||
                layer_sizes.back() != number_labels || number_labels == 0)
                throw std::invalid_argument("Ungueltige Layer-, Batch- oder Labelgroesse");

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
                Layers.emplace_back(layer_sizes[index - 1], layer_sizes[index], batch_size, *activation_, optimizer_config);
                if (index + 1 < layer_sizes.size()) {
                    if (activation == ActivationType::ReLU)
                        activations.push_back(std::make_unique<ReLU>());
                    else
                        activations.push_back(std::make_unique<Sigmoid>());
                } else {
                    activations.push_back(nullptr);
                }
            }

        }

        //trainiert das Neuronale Netz
        void train(const std::vector<std::vector<float>>& train_data,
                   const std::vector<int>& train_labels,
                   const std::vector<std::vector<float>>& validation_data,
                   const std::vector<int>& validation_labels);

        // Bestimmt den Loss auf dem Testdatensatz
        void test(const std::vector<std::vector<float>>&, const std::vector<int>&);


        // Gibt eine Printe Ausgabe für den(zufälligen) Batch
        void forward_test();

    private:

        std::vector<Layer<OptimizerType>> Layers;
        std::vector<std::unique_ptr<ActivationFunction>> activations;

        std::unique_ptr<ActivationFunction> activation_;
        std::unique_ptr<LossFunction> loss_;
        size_t number_labels;
        const size_t batch_size;
        LossType loss_type;
        size_t input_size;
        size_t current_size;
        typename OptimizerType::Config optimizer_config;

        int epochs_without_improvement;
        bool trained = false;

        // Gibt den Loss vom Batch an
        float forward(const Matrix<Layout::ColumnMajor>& input, const std::vector<int>& labels, Mode);

        // Berechnet die Ableitungen und passt die Gewichte an
        void backward(const Matrix<Layout::ColumnMajor>& input, const std::vector<int>& labels);

        // Berechnet einen Forward und misst dabei die precision und Zeit
        test_result test_run(const std::vector<std::vector<float>>& data, const std::vector<int>& y);
};

#include "neural_network.cpp"
