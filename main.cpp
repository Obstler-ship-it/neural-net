#include "mnist_reader/reader.hpp"
#include "src/neural_network.hpp"
#include "src/optimizers/momentum.hpp"
#include "src/optimizers/adam.hpp"
#include "src/optimizers/sgd.hpp"
#include "src/optimizers/rms_prop.hpp"
#include <chrono>
#include <print>

void start(){
    
    // Optimizers Adam, SGD, Momentum, RMSprop
    // Input 784, batch_size 64 and early stop 3
    // Activationfunctions are Sigmoid and ReLU
    // Last Dense-Layer must be size 10
    // Lossfunctions are CrossEntropy and MSE

    NeuralNetwork<Momentum> net(784, 64, 3); 
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
    
    const size_t seed = 19751979;
    set_random_seed(seed);

    const auto start_time = std::chrono::steady_clock::now();
    std::println("Started ...");

    start();

    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start_time).count();
    std::println("Time {}m{}.{:03}s", elapsed / 60000,
                 (elapsed / 1000) % 60, elapsed % 1000);
}
