#pragma once

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
#include <stdexcept>
#include <vector>

template<class Optimizer>
float NeuralNetwork<Optimizer>::forward(const Matrix<Layout::ColumnMajor>& input, const std::vector<int>& y, Mode mod){
    if (Layers.empty())
        throw std::logic_error("Netz benoetigt mindestens einen Dense-Layer");
    for (size_t i=0; i < Layers.size(); ++i) {
        const auto& previous = i == 0 ? input : Layers[i-1].a;
        if (activations[i])
            Layers[i].forward(previous, *activations[i]);
        else
            Layers[i].forward(previous);
    }
    softmax::forward(Layers.back().a);

    if (mod == Mode::train or mod == Mode::test)
        return loss_->forward(Layers.back().a, y);
    else if (mod == Mode::test_full)
        return ARG_MAX::count(Layers.back().a, y, number_labels);
    else
        throw std::logic_error("Gibt keinen Mode!");
}

template<class Optimizer>
void NeuralNetwork<Optimizer>::backward(const Matrix<Layout::ColumnMajor>& input, const std::vector<int>& y){
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
    for (size_t i=Layers.size(); i > 1; --i) {
        const size_t index = i - 1;
        if (activations[index])
            gradient = Layers[index].backward(gradient, *activations[index], Layers[index-1].a);
        else
            gradient = Layers[index].backward(gradient, Layers[index-1].a);
    }
    if (activations[0])
        Layers[0].backward(gradient, input, *activations[0]);
    else
        Layers[0].backward(gradient, input);
}

template<class Optimizer>
void NeuralNetwork<Optimizer>::train(
    const std::vector<std::vector<float>>& train_data,
    const std::vector<int>& train_labels,
    const std::vector<std::vector<float>>& validation_data,
    const std::vector<int>& validation_labels)
{

    if (train_labels.size() != train_data.size())
        throw std::invalid_argument("Unterschiedliche Anzahl Labels und Datensätze");
    if (train_data.empty() || batch_size == 0 || train_data.size() < batch_size)
        throw std::invalid_argument("Datensatz ist leer oder kleiner als ein Batch");
    if (Layers.empty())
        throw std::logic_error("Netz benoetigt mindestens einen Dense-Layer");

    std::vector<size_t> indizes(train_data.size());
    std::iota(indizes.begin(), indizes.end(), 0);
    size_t rows = train_data[0].size();
    size_t count = 0;
    size_t epoch = 0;
    float precision = 0;
    float best_precision = -1.0f;
    int no_improvement = 0;

    do{
        ++epoch;
        std::shuffle(indizes.begin(), indizes.end(), random_generator());
        size_t count_loss = 0;
        float loss = 0;
        double batch_build_ms = 0.0;
        double forward_ms = 0.0;
        double backward_ms = 0.0;

        for (size_t i=0; i + batch_size <= train_data.size(); i += batch_size) {

            const auto batch_start = std::chrono::steady_clock::now();
            count += batch_size;
            ++count_loss;
            std::vector<int> y(batch_size);
            Matrix<Layout::ColumnMajor> input(rows, batch_size, Initialization::Uninitialized);

            for (size_t j=0; j < batch_size; ++j) {
                size_t idx = indizes[i+j];
                y[j] = train_labels[idx];
                input.data.insert(input.data.end(), train_data[idx].begin(), train_data[idx].end());
            }
            const auto batch_end = std::chrono::steady_clock::now();
            batch_build_ms += std::chrono::duration<double, std::milli>(batch_end - batch_start).count();

            const auto forward_start = std::chrono::steady_clock::now();
            loss +=  forward(input, y, Mode::train) / batch_size;
            const auto forward_end = std::chrono::steady_clock::now();
            forward_ms += std::chrono::duration<double, std::milli>(forward_end - forward_start).count();

            const auto backward_start = std::chrono::steady_clock::now();
            backward(input, y);
            const auto backward_end = std::chrono::steady_clock::now();
            backward_ms += std::chrono::duration<double, std::milli>(backward_end - backward_start).count();
        }

        std::println("Epoch {} timing [ms]: batch {:.1f}, forward {:.1f}, backward {:.1f}; digits {}, loss {:.3f}",
            epoch, batch_build_ms, forward_ms, backward_ms, batch_size * count_loss, loss/count_loss);

        auto[validation_precision, ms, size] = test_run(validation_data, validation_labels);
        precision = validation_precision;

        if (precision > best_precision + 0.001f) {
            best_precision = precision;
        }
        else
            ++no_improvement;

        std::println("Epoch {} validation:, precision {:.4f}, forward {:.1f}, no_improvements {}, digits {}",
            epoch, precision, ms, no_improvement, size);

    } while (no_improvement <= epochs_without_improvement);

    std::println("Training finished after {} epochs ({} total samples).", epoch, count);
}

template<class Optimizer>
test_result NeuralNetwork<Optimizer>::test_run(const std::vector<std::vector<float>>& data, const std::vector<int>& y){

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

    return test_result(precision, ms, y.size());
}

template<class Optimizer>
void NeuralNetwork<Optimizer>::test(const std::vector<std::vector<float>>& data, const std::vector<int>& y){
    auto[precision, ms, size] = test_run(data, y);
    std::println("Precission: {:.2f}% in {}ms for {} Digits", precision * 100, ms, size);
}
