#include "mnist_reader/reader.hpp"
#include "src/neural_network.hpp"

int main(){
    const size_t seed = 163;
    set_random_seed(seed);

    auto [input_train, labels_train, train_rows, train_columns] = get_training_data();

    NeuralNetwork net({784, 1024, 256, 128, 32, 16, 10}, 10, ActivationType::ReLU, LossType::CrossEntropy, 64, 0.075);
    net.train(input_train, labels_train);

    auto [input_test, labels_test, test_rows, test_columns] = get_test_data();
    net.test(input_test, labels_test);
}
