// Hat Codex geschrieben
#include "reader.hpp"
#include "../src/random/random.hpp"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using Images = std::vector<std::vector<float>>;

std::uint32_t read_u32_big_endian(std::ifstream& file) {
    std::uint8_t bytes[4];
    file.read(reinterpret_cast<char*>(bytes), sizeof(bytes));

    if (!file) {
        throw std::runtime_error("Could not read MNIST file header");
    }

    return (static_cast<std::uint32_t>(bytes[0]) << 24) |
           (static_cast<std::uint32_t>(bytes[1]) << 16) |
           (static_cast<std::uint32_t>(bytes[2]) << 8) |
           static_cast<std::uint32_t>(bytes[3]);
}

MnistDataset load_dataset(
    const std::string& image_path,
    const std::string& label_path) {
    std::ifstream image_file(image_path, std::ios::binary);
    std::ifstream label_file(label_path, std::ios::binary);

    if (!image_file || !label_file) {
        throw std::runtime_error(
            "Could not open MNIST files: " + image_path + " and " + label_path);
    }

    const std::uint32_t image_magic = read_u32_big_endian(image_file);
    const std::uint32_t image_count = read_u32_big_endian(image_file);
    const std::uint32_t rows = read_u32_big_endian(image_file);
    const std::uint32_t columns = read_u32_big_endian(image_file);
    const std::uint32_t label_magic = read_u32_big_endian(label_file);
    const std::uint32_t label_count = read_u32_big_endian(label_file);

    if (image_magic != 2051 || label_magic != 2049) {
        throw std::runtime_error("Invalid MNIST file format");
    }

    if (rows != 28 || columns != 28 || image_count != label_count) {
        throw std::runtime_error("Unexpected MNIST dimensions or label count");
    }

    Images images(image_count);
    for (auto& image : images) {
        image.resize(rows * columns);
    }
    std::vector<int> labels(label_count);

    for (auto& image : images) {
        for (float& pixel : image) {
            std::uint8_t value;
            image_file.read(reinterpret_cast<char*>(&value), sizeof(value));

            if (!image_file) {
                throw std::runtime_error("Could not read MNIST image data");
            }

            pixel = static_cast<float>(value) / 255.0f;
        }
    }

    for (int& label : labels) {
        std::uint8_t value;
        label_file.read(reinterpret_cast<char*>(&value), sizeof(value));

        if (!label_file) {
            throw std::runtime_error("Could not read MNIST label data");
        }

        label = static_cast<int>(value);
    }

    return {std::move(images), std::move(labels)};
}

} // namespace

MnistDataset get_training_data() {
    return load_dataset(
        MNIST_DATA_DIR "/train-images.idx3-ubyte",
        MNIST_DATA_DIR "/train-labels.idx1-ubyte");
}

MnistDataset get_test_data() {
    return load_dataset(
        MNIST_DATA_DIR "/t10k-images.idx3-ubyte",
        MNIST_DATA_DIR "/t10k-labels.idx1-ubyte");
}

std::pair<MnistDataset, MnistDataset> split_dataset(MnistDataset dataset) {
    std::vector<size_t> indices(dataset.images.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), random_generator());
    const size_t split = indices.size() * 9 / 10;

    MnistDataset train;
    MnistDataset valid;
    train.images.reserve(split);
    train.labels.reserve(split);
    valid.images.reserve(indices.size() - split);
    valid.labels.reserve(indices.size() - split);

    for (size_t position = 0; position < indices.size(); ++position) {
        const size_t index = indices[position];
        auto& target = position < split ? train : valid;
        target.images.push_back(std::move(dataset.images[index]));
        target.labels.push_back(dataset.labels[index]);
    }

    return {std::move(train), std::move(valid)};
}
