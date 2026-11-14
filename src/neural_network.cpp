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
    MaxPool2D::output_size(output_rows, output_columns, convolution.NUMBER_KERNELS);
    const size_t pooled_rows = output_rows / 2;
    const size_t pooled_columns = output_columns / 2;
    if (pooled_rows < second_convolution.KERNEL_SIZE || pooled_columns < second_convolution.KERNEL_SIZE)
        throw std::invalid_argument("Feature-Maps sind kleiner als der zweite Kernel");
    if (MaxPool2D::output_size(pooled_rows - second_convolution.KERNEL_SIZE + 1,
                              pooled_columns - second_convolution.KERNEL_SIZE + 1,
                              second_convolution.NUMBER_KERNELS) != Layers.front().W.columns)
        throw std::invalid_argument("Pooling-Ausgabe passt nicht zur ersten Dense-Schicht");
}

float NeuralNetwork::forward(const Images& input, std::span<const size_t> batch_indices,
                             const std::vector<int>& labels, Mode mode) {
    TimingProfiler::Scope total_timer(profiler_, "forward.total");
    backward_ready_ = false;
    validate_images(input);
    validate_labels(labels, batch_indices.size());
    if (mode != Mode::train && mode != Mode::test && mode != Mode::test_full)
        throw std::invalid_argument("Ungültiger Modus");

    const size_t first_output_rows = input.rows_per_image - convolution.KERNEL_SIZE + 1;
    const size_t first_output_columns = input.columns_per_image - convolution.KERNEL_SIZE + 1;
    const auto& convolution_output = measure("forward.conv1_mec", [&]() -> const Matrix<Layout::ColumnMajor>& {
        return convolution.forward(input, batch_indices, *activation_);
    });
    measure("forward.pool1", [&] {
        first_pooled_output_ = first_pooling_.forward(convolution_output,
            first_output_rows, first_output_columns, convolution.NUMBER_KERNELS);
    });
    const auto& second_output = measure("forward.conv2", [&]() -> const Matrix<Layout::ColumnMajor>& {
        return second_convolution.forward(first_pooled_output_,
            first_output_rows / 2, first_output_columns / 2, *activation_);
    });
    measure("forward.pool2", [&] {
        pooled_output_ = pooling_.forward(second_output,
            first_output_rows / 2 - second_convolution.KERNEL_SIZE + 1,
            first_output_columns / 2 - second_convolution.KERNEL_SIZE + 1, second_convolution.NUMBER_KERNELS);
    });
    input_size_ = input.rows_per_image * input.columns_per_image;
    // Auch beim letzten Teilbatch müssen Dense- und Convolution-Gradienten gleich normiert werden.
    for (auto& layer : Layers)
        layer.batch_size = batch_indices.size();

    for (size_t index = 0; index < Layers.size(); ++index)
        measure(forward_stage_names_[index], [&] {
            const auto& previous = index == 0 ? pooled_output_ : Layers[index - 1].a;
            if (index + 1 == Layers.size()) Layers[index].forward(previous);
            else Layers[index].forward(previous, *activation_);
        });
    measure("forward.softmax", [&] { softmax::forward(Layers.back().a); });
    backward_ready_ = mode == Mode::train;

    return measure("forward.metric", [&] {
        if (mode == Mode::test_full) return ARG_MAX::count(Layers.back().a, labels, number_labels);
        return loss_->forward(Layers.back().a, labels);
    });
}

void NeuralNetwork::backward(const Matrix<Layout::ColumnMajor>& input, const std::vector<int>& labels) {
    TimingProfiler::Scope total_timer(profiler_, "backward.total");
    if (!backward_ready_)
        throw std::logic_error("backward benötigt einen neuen forward-Aufruf im Trainingsmodus");
    const auto& y_hat = Layers.back().a;
    validate_labels(labels, y_hat.columns);
    if (input.rows != input_size_ || input.columns != y_hat.columns ||
        input.data.size() != input.rows * input.columns)
        throw std::invalid_argument("Rohbilder passen nicht zum letzten forward-Batch");
    backward_ready_ = false;

    auto gradient = measure("backward.loss_gradient", [&] {
        Matrix<Layout::ColumnMajor> result = y_hat;
        if (loss_type == LossType::CrossEntropy) {
            for (size_t column = 0; column < labels.size(); ++column) result(labels[column], column) -= 1.0f;
        } else {
            auto loss_gradient = loss_->backward(y_hat, labels);
            result = softmax::backward(loss_gradient, labels, y_hat);
        }
        return result;
    });
    for (size_t remaining = Layers.size(); remaining > 0; --remaining) {
        const size_t index = remaining - 1;
        measure(backward_stage_names_[index], [&] {
            const auto& previous = index == 0 ? pooled_output_ : Layers[index - 1].a;
            if (index + 1 == Layers.size()) gradient = Layers[index].backward(gradient, previous, learning_rate);
            else gradient = Layers[index].backward(gradient, *activation_, previous, learning_rate);
        });
    }
    measure("backward.pool2", [&] { gradient = pooling_.backward(gradient); });
    measure("backward.conv2", [&] { gradient = second_convolution.backward(gradient, *activation_, first_pooled_output_, learning_rate); });
    measure("backward.pool1", [&] { gradient = first_pooling_.backward(gradient); });
    measure("backward.conv1", [&] { convolution.backward(learning_rate, gradient, *activation_, input); });
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
        if (profiler_) profiler_->reset();
        size_t profiled_batches = 0;
        std::shuffle(indices.begin(), indices.end(), random_generator());
        size_t epoch_samples = 0;
        float loss = 0.0f;
        double batch_build_ms = 0.0, forward_ms = 0.0, backward_ms = 0.0;
        const size_t epoch_target = std::min(data.size(), max_samples - count);
        auto last_progress = std::chrono::steady_clock::now();
        std::println("Epoch {} startet: {} Bilder, Batchgröße {} ...", epoch, epoch_target, batch_size);
        std::fflush(stdout);

        for (size_t i = 0; i < data.size() && count < max_samples; i += batch_size) {
            TimingProfiler::Scope batch_timer(profiler_, "batch.total");
            TimingProfiler::Scope build_timer(profiler_, "batch.build");
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
            build_timer.finish();
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
            batch_timer.finish();
            ++profiled_batches;
            if (profiler_ && (profiled_batches % 100 == 0 || epoch_samples == epoch_target))
                profiler_->report("train", epoch, profiled_batches, epoch_samples);
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
    if (profiler_) profiler_->reset();
    size_t profiled_batches = 0;
    std::println("Teste {} Bilder in Batches ...", data.number_images);
    std::fflush(stdout);
    const auto test_start = std::chrono::steady_clock::now();
    auto last_progress = test_start;
    double correct = 0.0;
    std::vector<size_t> indices;
    std::vector<int> batch_labels;
    for (size_t i = 0; i < data.number_images; i += batch_size) {
        TimingProfiler::Scope batch_timer(profiler_, "batch.total");
        TimingProfiler::Scope build_timer(profiler_, "batch.build");
        const size_t current_size = std::min(batch_size, data.number_images - i);
        indices.resize(current_size);
        std::iota(indices.begin(), indices.end(), i);
        batch_labels.assign(labels.begin() + i, labels.begin() + i + current_size);
        build_timer.finish();
        correct += forward(data, indices, batch_labels, Mode::test_full) * current_size;
        batch_timer.finish();
        ++profiled_batches;
        if (profiler_ && (profiled_batches % 100 == 0 || i + current_size == data.number_images))
            profiler_->report("test", 0, profiled_batches, i + current_size);
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
