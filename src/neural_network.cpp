#include "neural_network.hpp"
#include "softmax/softmax.hpp"
#include "loss/max.hpp"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cmath>
#include <numeric>
#include <print>
#include <stdexcept>

void NeuralNetwork::validate_labels(const std::vector<int>& labels, size_t count) const {
    if (labels.size() != count)
        throw std::invalid_argument("Unterschiedliche Anzahl Labels und Bilder");
    for (int label : labels)
        if (label < 0 || static_cast<size_t>(label) >= number_labels)
            throw std::invalid_argument("Label außerhalb der Klassen");
}

void NeuralNetwork::validate_images(const Images& input) const {
    const size_t kernel_size = convolution.KERNEL_SIZE;
    if (input.number_images == 0 || input.rows_per_image < kernel_size ||
        input.columns_per_image < kernel_size)
        throw std::invalid_argument("Ungültige Bilddimensionen");
    const size_t output_rows = input.rows_per_image - kernel_size + 1;
    const size_t output_columns = input.columns_per_image - kernel_size + 1;
    if (input.data.size() != input.number_images * output_rows * input.columns_per_image * kernel_size)
        throw std::invalid_argument("MEC-Daten passen nicht zu Bild- und Kernelgröße");
    if (MaxPool2D::output_size(output_rows, output_columns, convolution.NUMBER_KERNELS) != Layers.front().W.columns)
        throw std::invalid_argument("Pooling-Ausgabe passt nicht zur ersten Dense-Schicht");
}

float NeuralNetwork::forward(const Images& input, std::span<const size_t> batch_indices,
                             const std::vector<int>& labels, Mode mode) {
    backward_ready_ = false;
    validate_images(input);
    validate_labels(labels, batch_indices.size());
    if (mode != Mode::train && mode != Mode::test && mode != Mode::test_full)
        throw std::invalid_argument("Ungültiger Modus");

    const auto convolution_output = convolution.forward(input, batch_indices, *activation_);
    pooled_output_ = pooling_.forward(convolution_output,
        input.rows_per_image - convolution.KERNEL_SIZE + 1,
        input.columns_per_image - convolution.KERNEL_SIZE + 1, convolution.NUMBER_KERNELS);
    input_size_ = input.rows_per_image * input.columns_per_image;
    // Auch beim letzten Teilbatch müssen Dense- und Convolution-Gradienten gleich normiert werden.
    for (auto& layer : Layers)
        layer.batch_size = batch_indices.size();

    Layers.front().forward(pooled_output_, *activation_);
    for (size_t i = 1; i + 1 < Layers.size(); ++i)
        Layers[i].forward(Layers[i - 1].a, *activation_);
    Layers.back().forward(Layers[Layers.size() - 2].a);
    softmax::forward(Layers.back().a);
    backward_ready_ = mode == Mode::train;

    if (mode == Mode::test_full)
        return ARG_MAX::count(Layers.back().a, labels, number_labels);
    return loss_->forward(Layers.back().a, labels);
}

void NeuralNetwork::backward(const Matrix<Layout::ColumnMajor>& input, const std::vector<int>& labels) {
    if (!backward_ready_)
        throw std::logic_error("backward benötigt einen neuen forward-Aufruf im Trainingsmodus");
    const auto& y_hat = Layers.back().a;
    validate_labels(labels, y_hat.columns);
    if (input.rows != input_size_ || input.columns != y_hat.columns ||
        input.data.size() != input.rows * input.columns)
        throw std::invalid_argument("Rohbilder passen nicht zum letzten forward-Batch");
    backward_ready_ = false;

    Matrix<Layout::ColumnMajor> gradient = y_hat;
    if (loss_type == LossType::CrossEntropy) {
        for (size_t column = 0; column < labels.size(); ++column)
            gradient(labels[column], column) -= 1.0f;
    } else {
        auto loss_gradient = loss_->backward(y_hat, labels);
        gradient = softmax::backward(loss_gradient, labels, y_hat);
    }
    gradient = Layers.back().backward(gradient, Layers[Layers.size() - 2].a, learning_rate);
    for (size_t i = Layers.size() - 2; i > 0; --i)
        gradient = Layers[i].backward(gradient, *activation_, Layers[i - 1].a, learning_rate);
    gradient = Layers.front().backward(gradient, *activation_, pooled_output_, learning_rate);
    gradient = pooling_.backward(gradient);
    convolution.backward(learning_rate, gradient, *activation_, input);
}

void NeuralNetwork::train(const std::vector<std::vector<float>>& data, const Images& mec_data,
                          const std::vector<int>& labels, size_t max_samples) {
    validate_images(mec_data);
    validate_labels(labels, data.size());
    if (data.size() != mec_data.number_images || max_samples == 0)
        throw std::invalid_argument("Rohbilder und MEC-Daten müssen zusammenpassen; Trainingsumfang muss positiv sein");
    const size_t rows = mec_data.rows_per_image * mec_data.columns_per_image;
    for (const auto& image : data)
        if (image.size() != rows)
            throw std::invalid_argument("Rohbild hat falsche Größe");

    std::vector<size_t> indices(data.size());
    std::iota(indices.begin(), indices.end(), 0);
    size_t count = 0;
    size_t epoch = 0;
    while (count < max_samples) {
        ++epoch;
        std::shuffle(indices.begin(), indices.end(), random_generator());
        size_t epoch_samples = 0;
        float loss = 0.0f;
        double batch_build_ms = 0.0, forward_ms = 0.0, backward_ms = 0.0;
        const size_t epoch_target = std::min(data.size(), max_samples - count);
        auto last_progress = std::chrono::steady_clock::now();
        std::println("Epoch {} startet: {} Bilder, Batchgröße {} ...", epoch, epoch_target, batch_size);
        std::fflush(stdout);

        for (size_t i = 0; i < data.size() && count < max_samples; i += batch_size) {
            const auto batch_start = std::chrono::steady_clock::now();
            const size_t current_size = std::min({batch_size, data.size() - i, max_samples - count});
            const auto batch_indices = std::span<const size_t>(indices).subspan(i, current_size);
            std::vector<int> batch_labels;
            batch_labels.reserve(current_size);
            Matrix<Layout::ColumnMajor> input(rows, current_size, Initialization::Uninitialized);
            for (size_t index : batch_indices) {
                batch_labels.push_back(labels[index]);
                input.data.insert(input.data.end(), data[index].begin(), data[index].end());
            }
            const auto batch_end = std::chrono::steady_clock::now();
            batch_build_ms += std::chrono::duration<double, std::milli>(batch_end - batch_start).count();

            const auto forward_start = std::chrono::steady_clock::now();
            loss += forward(mec_data, batch_indices, batch_labels, Mode::train);
            const auto forward_end = std::chrono::steady_clock::now();
            forward_ms += std::chrono::duration<double, std::milli>(forward_end - forward_start).count();

            const auto backward_start = std::chrono::steady_clock::now();
            backward(input, batch_labels);
            const auto backward_end = std::chrono::steady_clock::now();
            backward_ms += std::chrono::duration<double, std::milli>(backward_end - backward_start).count();
            count += current_size;
            epoch_samples += current_size;
            if (i == 0 || epoch_samples == epoch_target ||
                backward_end - last_progress >= std::chrono::seconds(2)) {
                std::println("Epoch {}: {}/{} Bilder, Loss {:.3f}; letzter Batch {:.0f} ms",
                             epoch, epoch_samples, epoch_target, loss / epoch_samples,
                             std::chrono::duration<double, std::milli>(backward_end - batch_start).count());
                std::fflush(stdout);
                last_progress = backward_end;
            }
        }
        std::println("Epoch {} timing [ms]: batch {:.1f}, forward {:.1f}, backward {:.1f}; digits {}, loss {:.3f}",
                     epoch, batch_build_ms, forward_ms, backward_ms, epoch_samples, loss / epoch_samples);
        std::fflush(stdout);
    }
    trained = true;
}

void NeuralNetwork::test(const Images& data, const std::vector<int>& labels) {
    validate_images(data);
    validate_labels(labels, data.number_images);
    std::println("Teste {} Bilder in Batches ...", data.number_images);
    std::fflush(stdout);
    const auto test_start = std::chrono::steady_clock::now();
    auto last_progress = test_start;
    double correct = 0.0;
    std::vector<size_t> indices;
    std::vector<int> batch_labels;
    for (size_t i = 0; i < data.number_images; i += batch_size) {
        const size_t current_size = std::min(batch_size, data.number_images - i);
        indices.resize(current_size);
        std::iota(indices.begin(), indices.end(), i);
        batch_labels.assign(labels.begin() + i, labels.begin() + i + current_size);
        correct += forward(data, indices, batch_labels, Mode::test_full) * current_size;
        const auto now = std::chrono::steady_clock::now();
        if (now - last_progress >= std::chrono::seconds(2)) {
            std::println("Test: {}/{} Bilder", i + current_size, data.number_images);
            std::fflush(stdout);
            last_progress = now;
        }
    }
    const auto test_end = std::chrono::steady_clock::now();
    const auto ms = std::chrono::duration<float, std::milli>(test_end - test_start).count();
    std::println("Genauigkeit: {:.2f}% in {}ms für {} Ziffern", correct / data.number_images * 100, ms, labels.size());
    std::fflush(stdout);
}
