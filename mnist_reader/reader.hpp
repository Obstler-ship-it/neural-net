// Reader zum einlesen des MNIST Datensatzes
#include <vector>
#include <utility>

std::pair<std::vector<std::vector<float>>, std::vector<int>> get_training_data();

std::pair<std::vector<std::vector<float>>, std::vector<int>> get_test_data();
