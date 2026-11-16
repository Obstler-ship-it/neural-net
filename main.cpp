#include "matrix/matrix.cpp"
#include "mnist_reader/reader.hpp"
#include "src/neural_network.hpp"
#include "src/optimizers/adam.hpp"
#include "src/optimizers/sgd.hpp"
#include "src/optimizers/momentum.hpp"
#include "src/optimizers/rms_prop.hpp"
#include <chrono>
#include <print>

void start(){
    const size_t seed = 19751979;
    set_random_seed(seed);

    NeuralNetwork<Momentum> net(784, 64); //Optimizers Adam, SGD, Momentum, RMSprop
    net.add<Dense>(128);
    net.add<ReLU>();
    net.add<Dense>(32);
    net.add<ReLU>();
    net.add<Dense>(10);
    net.set_loss<CrossEntropy>();

    auto [train, valid] = split_dataset(get_training_data());
    net.train(train.images, train.labels, valid.images, valid.labels);

    auto [input_test, labels_test] = get_test_data();
    net.test(input_test, labels_test);
}

int main(){
    const auto start_time = std::chrono::steady_clock::now();
    std::println("Started ...");
    start();
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start_time).count();
    std::println("Time {}m{}.{:03}s", elapsed / 60000,
                 (elapsed / 1000) % 60, elapsed % 1000);
    return 0;
}

/* Irgendwie so etwas sollte man machen können
 * NeuralNet net;

     // Dimensionen hier explizit: Eingabekanäle, Ausgabekanäle, Kernelgröße
     net.add<Conv2D>(1, 32, 3);
     net.add<ReLU>();
     net.add<MaxPool2D>(2);

     net.add<Conv2D>(32, 64, 3);
     net.add<ReLU>();
     net.add<MaxPool2D>(2);

     net.add<Flatten>();
     net.add<Dense>(64 * 5 * 5, 10);

     Adam optimizer(0.001f);
     CrossEntropyLoss loss;
 */
