#include "mnist_reader/reader.hpp"
#include "src/neural_network.hpp"
#include "src/matrix/matrix.hpp"

int main(){
    const size_t seed = 163;
    set_random_seed(seed);
    auto[input_train, labels_train] = get_training_data();
    NeuralNetwork net({784, 64, 16, 10}, 10, ActivationType::ReLU, LossType::CrossEntropy, 64, 0.1);
    net.train(input_train, labels_train);
    auto[input_test, labels_test] = get_test_data();
    net.test(input_test, labels_test);
}
