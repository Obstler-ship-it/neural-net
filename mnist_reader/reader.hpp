// Reader zum einlesen des MNIST Datensatzes
#include <cstddef>
#include <vector>

struct MnistDataset {
    std::vector<std::vector<float>> images;
    std::vector<int> labels;
    std::size_t rows;
    std::size_t columns;
};

MnistDataset get_training_data();

MnistDataset get_test_data();
