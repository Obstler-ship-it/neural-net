#include "mnist_reader/reader.hpp"
#include "src/mec/mec.hpp"
#include "src/neural_network.hpp"
#include <cstdio>
#include <print>

int main(){
    const size_t seed = 163;
    set_random_seed(seed);
    constexpr size_t kernel_size = 3;
    constexpr size_t number_kernels = 8;

    std::println("Lade MNIST-Trainingsdaten ...");
    std::fflush(stdout);
    auto [input_train, labels_train, train_rows, train_columns] = get_training_data();
    if (train_rows < kernel_size || train_columns < kernel_size)
        throw std::invalid_argument("Bilder sind kleiner als der Kernel");
    const size_t features = MaxPool2D::output_size(train_rows - kernel_size + 1,
                                                   train_columns - kernel_size + 1, number_kernels);

    std::println("Initialisiere Netzwerk: {} Bilder, {} Features nach Max-Pooling pro Bild ...", input_train.size(), features);
    std::fflush(stdout);
    NeuralNetwork net({features, 1024, 256, 128, 32, 16, 10}, 10,
                      ActivationType::ReLU, LossType::CrossEntropy, 64, 0.025,
                      kernel_size, number_kernels);
    {
        std::println("Bereite MEC-Trainingsdaten vor ...");
        std::fflush(stdout);
        const auto mec_train = mec(input_train, train_rows, train_columns, kernel_size);
        net.train(input_train, mec_train, labels_train);
    }

    std::println("Lade MNIST-Testdaten ...");
    std::fflush(stdout);
    auto [input_test, labels_test, test_rows, test_columns] = get_test_data();
    std::println("Bereite MEC-Testdaten vor ...");
    std::fflush(stdout);
    const auto mec_test = mec(input_test, test_rows, test_columns, kernel_size);
    net.test(mec_test, labels_test);
}
