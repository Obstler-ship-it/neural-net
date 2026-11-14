#include "mnist_reader/reader.hpp"
#include "src/mec/mec.hpp"
#include "src/neural_network.hpp"
#include <cstdio>
#include <charconv>
#include <cstdlib>
#include <filesystem>
#include <print>
#include <string_view>

int main(){
    TimingProfiler profiler;
    if (const char* path = std::getenv("MNIST_PROFILE")) {
        profiler.enable(path);
        std::println("Profiling CSV: {}", std::filesystem::absolute(path).string());
    }
    const size_t seed = 165;
    set_random_seed(seed);
    constexpr size_t kernel_size = 3;
    constexpr size_t number_kernels = 32;
    constexpr size_t second_number_kernels = 64;

    std::println("Lade MNIST-Trainingsdaten ...");
    std::fflush(stdout);
    auto [input_train, labels_train, train_rows, train_columns] = profiler.measure("data.train.load", [] { return get_training_data(); });
    if (train_rows < 3 * kernel_size + 1 || train_columns < 3 * kernel_size + 1)
        throw std::invalid_argument("Bilder sind zu klein für zwei Convolution-Pooling-Blöcke");
    const size_t features = MaxPool2D::output_size((train_rows - kernel_size + 1) / 2 - kernel_size + 1,
                                                   (train_columns - kernel_size + 1) / 2 - kernel_size + 1, second_number_kernels);

    std::println("Initialisiere Netzwerk: {} Bilder, {} Features nach Max-Pooling pro Bild ...", input_train.size(), features);
    std::fflush(stdout);
    auto net = profiler.measure("network.initialize", [&] {
        return NeuralNetwork({features, 10}, 10,
                      ActivationType::ReLU, LossType::CrossEntropy, 64, 0.015,
                      kernel_size, number_kernels, second_number_kernels);
    });
    if (profiler.enabled()) net.set_profiler(profiler);
    {
        std::println("Bereite MEC-Trainingsdaten vor ...");
        std::fflush(stdout);
        const auto mec_train = profiler.measure("mec.train.prepare", [&] {
            return mec(input_train, train_rows, train_columns, kernel_size);
        });
        if (profiler.enabled())
            std::println("MEC train: Rohpixel {:.1f} MiB, MEC {:.1f} MiB, reserviert {:.1f} MiB",
                input_train.size() * train_rows * train_columns * sizeof(float) / 1048576.0,
                mec_train.data.size() * sizeof(float) / 1048576.0,
                mec_train.data.capacity() * sizeof(float) / 1048576.0);
        profiler.report("setup_train", 0, 0, input_train.size());
        if (const char* setting = std::getenv("MNIST_TRAIN_SAMPLES")) {
            const std::string_view value(setting);
            size_t samples = 0;
            const auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), samples);
            if (error != std::errc{} || end != value.data() + value.size() || samples == 0)
                throw std::invalid_argument("MNIST_TRAIN_SAMPLES must be a positive integer");
            net.train(input_train, mec_train, labels_train, samples);
        } else {
            net.train(input_train, mec_train, labels_train);
        }
    }

    std::println("Lade MNIST-Testdaten ...");
    std::fflush(stdout);
    profiler.reset();
    auto [input_test, labels_test, test_rows, test_columns] = profiler.measure("data.test.load", [] { return get_test_data(); });
    std::println("Bereite MEC-Testdaten vor ...");
    std::fflush(stdout);
    const auto mec_test = profiler.measure("mec.test.prepare", [&] { return mec(input_test, test_rows, test_columns, kernel_size); });
    profiler.report("setup_test", 0, 0, input_test.size());
    net.test(mec_test, labels_test);
}
