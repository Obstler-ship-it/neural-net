#pragma once

#include "../matrix/matrix.hpp"
#include <vector>

class Momentum {
public:
    struct Config {
        float learning_rate = 0.01f;
        float momentum = 0.9f;
    };

    Momentum() = default;
    explicit Momentum(const Config& config) : config(config) {}

    // Ein Optimizer pro Layer mit gleichbleibenden Matrixgroessen.
    // Die Updates werden anschliessend von Gewichten und Bias subtrahiert.
    void step(Matrix<Layout::ColumnMajor>& dW, Matrix<Layout::ColumnMajor>& db) {
        update(dW.data, velocity_weights);
        update(db.data, velocity_bias);
    }

private:
    void update(std::vector<float>& gradient, std::vector<float>& velocity) {
        if (velocity.empty())
            velocity.resize(gradient.size(), 0.0f);

        for (size_t i = 0; i < gradient.size(); ++i) {
            velocity[i] = config.momentum * velocity[i] + gradient[i];
            gradient[i] = config.learning_rate * velocity[i];
        }
    }

    Config config;
    std::vector<float> velocity_weights, velocity_bias;
};
