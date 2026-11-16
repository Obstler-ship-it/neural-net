// Reader zum einlesen des MNIST Datensatzes
#include <vector>
#include <utility>

struct MnistDataset {
    std::vector<std::vector<float>> images;
    std::vector<int> labels;
};

MnistDataset get_training_data();

MnistDataset get_test_data();

std::pair<MnistDataset, MnistDataset> split_dataset(MnistDataset dataset);
