#include "mnist_reader/reader.hpp"
#include "src/neural_network.hpp"

int main(){
    const size_t seed = 163;
    set_random_seed(seed);

    auto [input_train, labels_train, train_rows, train_columns] = get_training_data();

    NeuralNetwork net({784, 64, 32, 16, 10}, 10, ActivationType::ReLU, LossType::CrossEntropy, 64, 0.075);
    net.train(input_train, labels_train);

    auto [input_test, labels_test, test_rows, test_columns] = get_test_data();
    net.test(input_test, labels_test);
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
