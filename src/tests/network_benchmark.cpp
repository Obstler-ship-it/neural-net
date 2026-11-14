#include "../neural_network.hpp"
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string_view>

namespace {
using Clock = std::chrono::steady_clock;

double milliseconds(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

void report(std::string_view name, std::vector<double> samples) {
    std::sort(samples.begin(), samples.end());
    const double median = (samples[(samples.size() - 1) / 2] + samples[samples.size() / 2]) / 2;
    std::cout << std::left << std::setw(24) << name << std::right << std::fixed
              << std::setprecision(3) << median << " ms/batch (median)\n";
}

template<typename Operation>
void measure(std::string_view name, size_t iterations, Operation operation) {
    std::vector<double> samples;
    float checksum = 0.0f;
    for (size_t iteration = 0; iteration < iterations + 2; ++iteration) {
        const auto start = Clock::now();
        checksum += operation();
        const auto end = Clock::now();
        if (iteration >= 2) samples.push_back(milliseconds(start, end));
    }
    report(name, samples);
    std::cout << "  Checksum: " << checksum << '\n';
}
}

int main(int argc, char* argv[]) {
    const size_t iterations = argc > 1 ? std::stoul(argv[1]) : 10;
    if (iterations == 0) throw std::invalid_argument("Iterations must be positive");
    constexpr size_t batch_size = 64;
    set_random_seed(164);
    NeuralNetwork net({800, 128, 32, 16, 10}, 10, ActivationType::ReLU,
                      LossType::CrossEntropy, batch_size, 0.025f, 3, 64, 32);
    std::vector<std::vector<float>> images(batch_size, std::vector<float>(28 * 28));
    Matrix<Layout::ColumnMajor> raw(28 * 28, batch_size, Initialization::Uninitialized);
    std::vector<size_t> indices(batch_size);
    std::iota(indices.begin(), indices.end(), 0);
    std::vector<int> labels(batch_size);
    for (size_t image = 0; image < batch_size; ++image) {
        for (float& pixel : images[image]) pixel = (random_float() + 1.0f) * 0.5f;
        raw.data.insert(raw.data.end(), images[image].begin(), images[image].end());
        labels[image] = image % 10;
    }
    const auto mec_start = Clock::now();
    const auto transformed = mec(images, 28, 28, 3);
    const auto mec_end = Clock::now();
    std::cout << "MEC preparation (once): " << milliseconds(mec_start, mec_end) << " ms, raw "
              << raw.data.size() * sizeof(float) / 1048576.0 << " MiB, MEC "
              << transformed.data.size() * sizeof(float) / 1048576.0 << " MiB\n";
    std::cout << "Batch 64, Conv 64 -> Pool -> Conv 32 -> Pool, " << iterations << " measured iterations, 2 warmups\n";
    measure("Conv1 forward", iterations, [&] {
        return net.convolution.forward(transformed, indices, *net.activation_).data.front();
    });
    measure("Conv1 raw (no MEC)", iterations, [&] {
        return net.convolution.forward(raw, 28, 28, *net.activation_).data.front();
    });
    MaxPool2D first_pooling;
    const auto features = first_pooling.forward(net.convolution.forward(transformed, indices, *net.activation_), 26, 26, 64);
    measure("Pool1 forward", iterations, [&] {
        return first_pooling.forward(net.convolution.output(), 26, 26, 64).data.front();
    });
    Matrix<Layout::ColumnMajor> pooling_gradient(features.rows, features.columns, Initialization::Random);
    measure("Pool1 backward", iterations, [&] {
        return first_pooling.backward(pooling_gradient).data.front();
    });
    measure("Conv2 forward", iterations, [&] {
        return net.second_convolution.forward(features, 13, 13, *net.activation_).data.front();
    });
    Matrix<Layout::ColumnMajor> upstream(11 * 11 * 32, batch_size, Initialization::Random);
    measure("Conv2 backward", iterations, [&] {
        auto gradient = upstream;
        return net.second_convolution.backward(gradient, *net.activation_, features, 0.0f).data.front();
    });
    std::vector<double> forward_times, backward_times, total_times;
    float checksum = 0.0f;
    for (size_t iteration = 0; iteration < iterations + 2; ++iteration) {
        const auto start = Clock::now();
        checksum += net.forward(transformed, indices, labels, Mode::train);
        const auto middle = Clock::now();
        net.backward(raw, labels);
        const auto end = Clock::now();
        if (iteration >= 2) {
            forward_times.push_back(milliseconds(start, middle));
            backward_times.push_back(milliseconds(middle, end));
            total_times.push_back(milliseconds(start, end));
        }
    }
    report("Network forward", forward_times);
    report("Network backward", backward_times);
    report("Network total", total_times);
    std::cout << "Loss checksum: " << checksum << '\n';
}
