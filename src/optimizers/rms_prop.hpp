#pragma once

#include "../matrix/matrix.hpp"
#include <vector>
#include <cmath>

class RMSprop {
public:
    struct Config {
        float learning_rate = 0.001f;
        float decay = 0.99f;
        float epsilon = 1e-8f;
    };

    RMSprop() = default;
    explicit RMSprop(const Config& config) : config(config) {}

    // Ein Optimizer pro Layer mit gleichbleibenden Matrixgroessen.
    // Die Updates werden anschliessend von Gewichten und Bias subtrahiert.
    void step(Matrix<Layout::ColumnMajor>& dW, Matrix<Layout::ColumnMajor>& db) {
        update(dW.data, squares_weights);
        update(db.data, squares_bias);
    }

private:
    void update(std::vector<float>& gradient, std::vector<float>& squares) {
        if (squares.empty())
            squares.resize(gradient.size(), 0.0f);

        for (size_t i = 0; i < gradient.size(); ++i) {
            const float g = gradient[i];
            squares[i] = config.decay * squares[i] + (1 - config.decay) * g * g;
            gradient[i] = config.learning_rate * g / (std::sqrt(squares[i]) + config.epsilon);
        }
    }

    Config config;
    std::vector<float> squares_weights, squares_bias;
};
