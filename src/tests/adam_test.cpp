#include "../optimizers/adam.hpp"
#include <cassert>
#include <cmath>
#include <vector>

void check_updates(const Adam::Config& config) {
    Adam optimizer(config);
    Matrix<Layout::ColumnMajor> weights(3, 4, Initialization::Zero);
    Matrix<Layout::ColumnMajor> bias(3, 1, Initialization::Zero);
    std::vector<double> m(15, 0.0), v(15, 0.0);
    for (int step = 1; step <= 100; ++step) {
        auto fill_and_expect = [&](auto& gradient, size_t offset) {
            std::vector<double> expected(gradient.data.size());
            for (size_t i = 0; i < gradient.data.size(); ++i) {
                // Positive, negative und dauerhaft verschwindende Gradienten.
                const float g = i == 0 ? 0.0f : static_cast<float>(static_cast<int>((i + step) % 7) - 3) * 0.25f;
                gradient.data[i] = g;
                const size_t j = offset + i;
                m[j] = config.beta1 * m[j] + (1.0 - config.beta1) * g;
                v[j] = config.beta2 * v[j] + (1.0 - config.beta2) * g * g;
                expected[i] = config.learning_rate * (m[j] / (1 - std::pow(config.beta1, step)))
                    / (std::sqrt(v[j] / (1 - std::pow(config.beta2, step))) + config.epsilon);
            }
            return expected;
        };
        const auto expected_weights = fill_and_expect(weights, 0);
        const auto expected_bias = fill_and_expect(bias, 12);
        optimizer.step(weights, bias);
        for (size_t i = 0; i < weights.data.size(); ++i)
            assert(std::isfinite(weights.data[i]) && std::abs(weights.data[i] - expected_weights[i]) < 1e-6);
        for (size_t i = 0; i < bias.data.size(); ++i)
            assert(std::isfinite(bias.data[i]) && std::abs(bias.data[i] - expected_bias[i]) < 1e-6);
    }
    // Ein Optimizer darf nicht mit anders geformten Parametern weiterlaufen.
    weights.resize(4, 3);
    bool rejected = false;
    try { optimizer.step(weights, bias); }
    catch (const std::invalid_argument&) { rejected = true; }
    assert(rejected);
}

int main() {
    check_updates(Adam::Config{});
    check_updates(Adam::Config{0.001f, 0.9f, 0.999f});
    check_updates(Adam::Config{0.01f, 0.0f, 0.0f});
    // Der Default-Konstruktor muss seine Konfiguration selbst besitzen.
    Adam optimizer;
    Matrix<Layout::ColumnMajor> weights(2, 3, Initialization::Zero);
    Matrix<Layout::ColumnMajor> bias(2, 1, Initialization::Zero);
    optimizer.step(weights, bias);
    for (float value : weights.data) assert(value == 0.0f);
    for (float value : bias.data) assert(value == 0.0f);
}
