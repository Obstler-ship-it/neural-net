#pragma once

#include "../matrix/matrix.hpp"

class SGD {
public:
    struct Config {
        float learning_rate = 0.075f;
    };

    SGD() = default;
    explicit SGD(const Config& config) : config(config) {}

    void step(Matrix<Layout::ColumnMajor>& dW, Matrix<Layout::ColumnMajor>& db) {
        dW *= config.learning_rate;
        db *= config.learning_rate;
    }

private:
    Config config;
};
