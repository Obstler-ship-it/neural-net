// Reader zum einlesen des MNIST Datensatzes
#include <vector>
#include <array>
#include <utility>

std::pair<std::vector<std::array<double, 784>>, std::vector<int>> get_training_data();

std::pair<std::vector<std::array<double, 784>>, std::vector<int>> get_test_data();
