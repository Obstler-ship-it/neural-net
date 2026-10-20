#include "../neural_network.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <string>

namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
void close(float actual, float expected, const char* message, float tolerance = 0.002f) {
    if (!std::isfinite(actual) || std::abs(actual - expected) > tolerance * (1.0f + std::abs(expected)))
        throw std::runtime_error(std::string(message) + ": " + std::to_string(actual)
                                 + " != " + std::to_string(expected));
}
template<class Exception, class Function>
void expect_throw(Function function) {
    try { function(); }
    catch (const Exception&) { return; }
    throw std::runtime_error("Erwartete Ausnahme fehlt");
}
std::vector<std::vector<float>> images() {
    std::vector<std::vector<float>> result(5, std::vector<float>(12));
    for (size_t i = 0; i < result.size(); ++i)
        for (size_t p = 0; p < 12; ++p)
            result[i][p] = 0.01f * (1 + (p * p * 7 + p * 3 + i * 11) % 97) + 0.001f * p;
    return result;
}
void initialize(NeuralNetwork& net) {
    for (size_t i = 0; i < net.convolution.Kernels.data.size(); ++i)
        net.convolution.Kernels.data[i] = 0.03f * (static_cast<int>(i % 5) - 2);
    std::fill(net.convolution.Bias.data.begin(), net.convolution.Bias.data.end(), 0.6f);
    for (auto& layer : net.Layers) {
        for (size_t i = 0; i < layer.W.data.size(); ++i)
            layer.W.data[i] = 0.04f * (static_cast<int>(i % 7) - 3);
        std::fill(layer.b.data.begin(), layer.b.data.end(), 0.4f);
    }
}
std::vector<float*> parameters(NeuralNetwork& net) {
    std::vector<float*> result;
    for (float& value : net.convolution.Kernels.data) result.push_back(&value);
    for (float& value : net.convolution.Bias.data) result.push_back(&value);
    for (auto& layer : net.Layers) {
        for (float& value : layer.W.data) result.push_back(&value);
        for (float& value : layer.b.data) result.push_back(&value);
    }
    return result;
}
Matrix<Layout::ColumnMajor> raw_batch(const std::vector<std::vector<float>>& raw,
                                     std::span<const size_t> indices) {
    Matrix<Layout::ColumnMajor> result(12, indices.size(), Initialization::Uninitialized);
    for (size_t index : indices)
        result.data.insert(result.data.end(), raw[index].begin(), raw[index].end());
    return result;
}

void check_gradients(ActivationType activation, LossType loss) {
    NeuralNetwork net({2, 4, 2}, 2, activation, loss, 4, 0.1f, 2, 2);
    initialize(net);
    const auto raw = images();
    const auto transformed = mec(raw, 3, 4, 2);
    const std::vector<size_t> indices{2, 0, 2};
    const std::vector<int> labels{1, 0, 1};
    const auto input = raw_batch(raw, indices);
    auto params = parameters(net);
    std::vector<float> before, numerical;
    for (float* parameter : params) {
        const float saved = *parameter;
        before.push_back(saved);
        constexpr float epsilon = 0.002f;
        *parameter = saved + epsilon;
        const float plus = net.forward(transformed, indices, labels, Mode::test);
        *parameter = saved - epsilon;
        const float minus = net.forward(transformed, indices, labels, Mode::test);
        *parameter = saved;
        numerical.push_back((plus - minus) / (2 * epsilon * indices.size()));
    }
    net.forward(transformed, indices, labels, Mode::train);
    net.backward(input, labels);
    // Dense-Matrixoperatoren können den Speicher ersetzen; Zeiger danach neu abfragen.
    params = parameters(net);
    for (size_t i = 0; i < params.size(); ++i)
        close((before[i] - *params[i]) / net.learning_rate, numerical[i], "Netzwerkgradient");
    require(std::abs(before[0] - *params[0]) > 1e-7f, "Convolution-Kernel wurde nicht trainiert");
    expect_throw<std::logic_error>([&] { net.backward(input, labels); });
}

void check_training_and_test() {
    NeuralNetwork trained({2, 4, 2}, 2, ActivationType::Sigmoid, LossType::CrossEntropy, 3, 0.1f, 2, 2);
    NeuralNetwork manual({2, 4, 2}, 2, ActivationType::Sigmoid, LossType::CrossEntropy, 3, 0.1f, 2, 2);
    initialize(trained);
    initialize(manual);
    const auto raw = images();
    const auto transformed = mec(raw, 3, 4, 2);
    const std::vector<int> labels{0, 1, 1, 0, 1};
    set_random_seed(123);
    trained.train(raw, transformed, labels, 5);
    require(trained.trained, "Trainingsstatus fehlt");
    set_random_seed(123);
    std::vector<size_t> order{0, 1, 2, 3, 4};
    std::shuffle(order.begin(), order.end(), random_generator());
    for (size_t start = 0; start < order.size(); start += 3) {
        const auto indices = std::span<const size_t>(order).subspan(start, std::min(size_t{3}, order.size() - start));
        std::vector<int> batch_labels;
        for (size_t index : indices) batch_labels.push_back(labels[index]);
        manual.forward(transformed, indices, batch_labels, Mode::train);
        manual.backward(raw_batch(raw, indices), batch_labels);
    }
    auto actual = parameters(trained), expected = parameters(manual);
    for (size_t i = 0; i < actual.size(); ++i)
        close(*actual[i], *expected[i], "Trainingsbatch", 1e-6f);

    std::vector<float> before;
    for (float* value : actual) before.push_back(*value);
    trained.test(transformed, labels);
    require(trained.Layers.back().a.columns == 2, "Test-Teilbatch fehlt");
    const std::vector<size_t> last{3, 4};
    manual.forward(transformed, last, {0, 1}, Mode::test_full);
    for (size_t i = 0; i < trained.Layers.back().a.data.size(); ++i)
        close(trained.Layers.back().a.data[i], manual.Layers.back().a.data[i], "Testausgabe", 1e-6f);
    actual = parameters(trained);
    for (size_t i = 0; i < actual.size(); ++i)
        close(*actual[i], before[i], "Test darf Gewichte nicht ändern", 1e-6f);
    expect_throw<std::logic_error>([&] { trained.backward(raw_batch(raw, last), {0, 1}); });

    // Auch Datensätze kleiner als batch_size und ein begrenztes Trainingsbudget funktionieren.
    const std::vector<std::vector<float>> single{raw[0]};
    trained.train(single, mec(single, 3, 4, 2), {0}, 1);
    require(trained.Layers.back().a.columns == 1, "Einzelbildbatch fehlt");
}

void check_pooling() {
    MaxPool2D pool;
    Matrix<Layout::ColumnMajor> gradient(4, 2, Initialization::Zero);
    expect_throw<std::logic_error>([&] { pool.backward(gradient); });
    Matrix<Layout::ColumnMajor> input(30, 2, Initialization::Zero);
    const std::vector<float> pixels{
        1,-4, 9,-2, 2,-8, 3,-7, 999,999,
        4,-3, 5,-2, 8,-6, 7,-9, 999,999,
        999,999, 999,999, 999,999, 999,999, 999,999};
    for (size_t b = 0; b < 2; ++b)
        for (size_t i = 0; i < 30; ++i) input(i,b) = pixels[i] + 10 * b;
    const auto output = pool.forward(input, 3, 5, 2);
    require(output.rows == 4 && output.columns == 2, "Pooling-Dimensionen");
    const std::vector<float> expected{9,-2,8,-6,19,8,18,4};
    for (size_t i = 0; i < expected.size(); ++i) {
        close(output.data[i], expected[i], "Pooling-Ausgabe", 0);
        gradient.data[i] = i + 1;
    }
    const auto backward = pool.backward(gradient);
    Matrix<Layout::ColumnMajor> expected_gradient(30, 2, Initialization::Zero);
    const std::vector<size_t> winners{2,3,14,15,32,33,44,45};
    for (size_t i = 0; i < winners.size(); ++i) expected_gradient.data[winners[i]] = i + 1;
    for (size_t i = 0; i < backward.data.size(); ++i)
        close(backward.data[i], expected_gradient.data[i], "Pooling-Gradient", 0);
    gradient.rows = 3;
    expect_throw<std::invalid_argument>([&] { pool.backward(gradient); });
    expect_throw<std::invalid_argument>([&] { pool.forward(input, 1, 15, 2); });
    require(MaxPool2D::output_size(26,26,8) == 1352, "MNIST-Pooling-Größe");
}

void check_validation() {
    NeuralNetwork net({2, 4, 2}, 2, ActivationType::Sigmoid, LossType::CrossEntropy, 3, 0.1f, 2, 2);
    const auto raw = images();
    auto transformed = mec(raw, 3, 4, 2);
    const std::vector<size_t> indices{0};
    expect_throw<std::invalid_argument>([&] { net.forward(transformed, indices, {2}, Mode::train); });
    expect_throw<std::invalid_argument>([&] { net.forward(transformed, indices, {}, Mode::train); });
    expect_throw<std::out_of_range>([&] { net.forward(transformed, std::vector<size_t>{5}, {0}, Mode::train); });
    net.forward(transformed, indices, {0}, Mode::train);
    Matrix<Layout::ColumnMajor> wrong(11, 1, Initialization::Zero);
    expect_throw<std::invalid_argument>([&] { net.backward(wrong, {0}); });
    transformed.data.pop_back();
    expect_throw<std::invalid_argument>([&] { net.forward(transformed, indices, {0}, Mode::train); });
    NeuralNetwork wrong_features({13, 4, 2}, 2, ActivationType::ReLU, LossType::CrossEntropy, 3, 0.1f, 2, 2);
    expect_throw<std::invalid_argument>([&] { wrong_features.forward(mec(raw, 3, 4, 2), indices, {0}, Mode::train); });
}
}
int main() {
    for (auto activation : {ActivationType::Sigmoid, ActivationType::ReLU})
        for (auto loss : {LossType::CrossEntropy, LossType::MeanSquareError})
            check_gradients(activation, loss);
    check_pooling();
    check_training_and_test();
    check_validation();
    std::cout << "Alle Netzwerk-Convolution-Tests erfolgreich.\n";
}
