//
#include "neural_network.hpp"
#include "matrix/matrix.hpp"
#include "softmax/softmax.hpp"
#include "loss/max.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <print>
#include <random>
#include <stdexcept>
#include <vector>


float NeuralNetwork::forward(const Matrix<Layout::ColumnMajor>& input, const std::vector<int>& y, Mode mod){
    Layers[0].forward(input, *activation_);
    for (size_t i=1; i < Layers.size() - 1; i++){
        Layers[i].forward(Layers[i-1].a, *activation_);
    }
    Layers[Layers.size() - 1].forward(Layers[Layers.size() - 2].a);
    softmax::forward(Layers.back().a);

    if (mod == Mode::train or mod == Mode::test)
        return loss_->forward(Layers.back().a, y);
    else if (mod == Mode::test_full)
        return ARG_MAX::count(Layers.back().a, y, number_labels);
    else
        throw std::logic_error("Gibt keinen Mode!");
}

void NeuralNetwork::backward(const Matrix<Layout::ColumnMajor>& input, const std::vector<int>& y){
    auto& y_hat = Layers.back().a;
    Matrix<Layout::ColumnMajor> gradient = y_hat;
    if (loss_type == LossType::CrossEntropy) {
        for (size_t column = 0; column < y.size(); ++column) {
            gradient(y[column], column) -= 1.0f;
        }
    } else {
        auto loss_gradient = loss_->backward(y_hat, y);
        gradient = softmax::backward(loss_gradient, y, y_hat);
    }
    gradient = Layers[Layers.size() - 1].backward(gradient, Layers[Layers.size() - 2].a, learning_rate);
    for (size_t i=Layers.size() - 2; i > 0; i--){
        gradient = Layers[i].backward(gradient, *activation_, Layers[i-1].a, learning_rate);
    }
    Layers[0].backward(gradient, *activation_, input, learning_rate);
}

void NeuralNetwork::train(const std::vector<std::vector<float>>& data, const std::vector<int>& labels){

    if (labels.size() != data.size())
        throw std::invalid_argument("Unterschiedliche Anzahl Labels und Datensätze");
    if (data.empty() || batch_size == 0 || data.size() < batch_size)
        throw std::invalid_argument("Datensatz ist leer oder kleiner als ein Batch");

    std::mt19937 generator{42};
    std::vector<size_t> indizes(data.size());
    std::iota(indizes.begin(), indizes.end(), 0);
    size_t rows = data[0].size();
    size_t count = 0;
    size_t epoch = 0;

    do{
        ++epoch;
        std::shuffle(indizes.begin(), indizes.end(), generator);
        size_t count_loss = 0;
        float loss = 1;
        double batch_build_ms = 0.0;
        double forward_ms = 0.0;
        double backward_ms = 0.0;

        for (size_t i=0; i + batch_size <= data.size() && count < 12000 * 64; i += batch_size) {

            const auto batch_start = std::chrono::steady_clock::now();
            count += batch_size;
            ++count_loss;
            std::vector<int> y(batch_size);
            Matrix<Layout::ColumnMajor> input(rows, batch_size, Initialization::Uninitialized);

            for (size_t j=0; j < batch_size; ++j) {
                size_t idx = indizes[i+j];
                y[j] = labels[idx];
                input.data.insert(input.data.end(), data[idx].begin(), data[idx].end());
            }
            const auto batch_end = std::chrono::steady_clock::now();
            batch_build_ms += std::chrono::duration<double, std::milli>(batch_end - batch_start).count();

            const auto forward_start = std::chrono::steady_clock::now();
            loss +=  forward(input, y, Mode::train) / batch_size;
            const auto forward_end = std::chrono::steady_clock::now();
            forward_ms += std::chrono::duration<double, std::milli>(forward_end - forward_start).count();

            // if (!std::isfinite(loss))
            //     throw std::runtime_error("Loss ist nicht endlich");

            const auto backward_start = std::chrono::steady_clock::now();
            backward(input, y);
            const auto backward_end = std::chrono::steady_clock::now();
            backward_ms += std::chrono::duration<double, std::milli>(backward_end - backward_start).count();
        }

        std::println("Epoch {} timing [ms]: batch {:.1f}, forward {:.1f}, backward {:.1f}; digits {}, loss {:.3f}",
            epoch, batch_build_ms, forward_ms, backward_ms, batch_size * count_loss, loss/count_loss);

    } while (count < 12000 * 64);

}

void NeuralNetwork::test(const std::vector<std::vector<float>>& data, const std::vector<int>& y){

    if (y.size() != data.size())
        throw std::invalid_argument("Unterschiedliche Anzahl Labels und Datensätze");
    if (data.empty())
        throw std::invalid_argument("Datensatz ist leer");

    const auto test_start = std::chrono::steady_clock::now();

    size_t rows = data[0].size();
    Matrix<Layout::ColumnMajor> input(rows, y.size(), Initialization::Uninitialized);

    for (size_t i=0; i < data.size(); ++i) {
        input.data.insert(input.data.end(), data[i].begin(), data[i].end());
    }

    float precision = forward(input, y, Mode::test_full);

    const auto test_end = std::chrono::steady_clock::now();
    auto ms = std::chrono::duration<float, std::milli>(test_end - test_start).count();
    std::println("Genauigkeit: {}% in {}ms für {} Ziffern", precision, ms, y.size());
}
