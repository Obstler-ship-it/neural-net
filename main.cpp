#include "matrix/matrix.cpp"
#include "mnist_reader/reader.hpp"
#include "src/neural_network.hpp"
#include "src/optimizers/adam.hpp"
#include "src/optimizers/sgd.hpp"
#include "src/optimizers/momentum.hpp"
#include <numeric>
#include <random>
#include <vector>

struct Dataset{
    std::vector<std::vector<float>> data;
    std::vector<int> labels;
};

std::pair<Dataset, Dataset> split(Dataset dataset){
    std::vector<size_t> indices(dataset.data.size());
    std::iota(indices.begin(), indices.end(), 0);

    std::shuffle(indices.begin(), indices.end(), random_generator());
    const size_t split = indices.size() * 9 / 10;

    Dataset train;
    Dataset valid;

    for (size_t position = 0; position < indices.size(); ++position) {
        const size_t index = indices[position];
        if (position < split) {
            train.data.push_back(std::move(dataset.data[index]));
            train.labels.push_back(dataset.labels[index]);
        } else {
            valid.data.push_back(std::move(dataset.data[index]));
            valid.labels.push_back(dataset.labels[index]);
        }
    }

    return {std::move(train), std::move(valid)};
}

void start(){
    const size_t seed = 163;
    set_random_seed(seed);

    auto [images, labels] = get_training_data();

    NeuralNetwork<Adam> net({784, 128, 32, 10}, 10, ActivationType::ReLU, LossType::CrossEntropy, 64);
    auto [train, valid] = split(Dataset{std::move(images), std::move(labels)});
    net.train(train.data, train.labels, valid.data, valid.labels);

    auto [input_test, labels_test] = get_test_data();
    net.test(input_test, labels_test);
    return;
}

int main(){
    start();
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
