#include "mnist_reader/reader.hpp"
#include "src/neural_network.hpp"

int main(){
    auto[input_train, labels_train] = get_training_data();
    NeuralNetwork net({784, 64, 16, 10}, 10, ActivationType::ReLU, LossType::CrossEntropy);
    net.train(input_train, labels_train);
    auto[input_test, labels_test] = get_test_data();
    net.test(input_test, labels_test);
}
