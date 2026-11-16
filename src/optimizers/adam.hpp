// Adam: Gradienten werden durch die zu subtrahierenden Updates ersetzt.
#pragma once

#include <cmath>
#include <stdexcept>
#include <vector>
#include "../matrix/matrix.hpp"

class Adam {
public:
    struct Config {
        float learning_rate = 0.001f;
        float beta1 = 0.9f;
        float beta2 = 0.999f;
        float epsilon = 1e-8f;
    };

    explicit Adam() : Adam(Config{}) {}

    explicit Adam(const Config& config) : config(config) {
        if (!(config.learning_rate > 0))
            throw std::invalid_argument("Adam: learning_rate must be > 0");
        if (!(config.beta1 >= 0 && config.beta1 < 1))
            throw std::invalid_argument("Adam: beta1 must be in [0, 1)");
        if (!(config.beta2 >= 0 && config.beta2 < 1))
            throw std::invalid_argument("Adam: beta2 must be in [0, 1)");
        if (!(config.epsilon > 0))
            throw std::invalid_argument("Adam: epsilon must be > 0");
    }

    void step(Matrix<Layout::ColumnMajor>& dW, Matrix<Layout::ColumnMajor>& db) {
        if (t == 0) {
            weights.initialize(dW);
            bias.initialize(db);
        }

        ++t;
        const float correction1 = config.learning_rate / (1 - std::pow(config.beta1, t));
        const float correction2 = 1.0f / (1 - std::pow(config.beta2, t));
        update(dW.data, weights, correction1, correction2);
        update(db.data, bias, correction1, correction2);
    }

private:
    struct State {
        std::vector<float> m;
        std::vector<float> v;
        size_t rows = 0;
        size_t columns = 0;

        void initialize(const Matrix<Layout::ColumnMajor>& gradient) {
            rows = gradient.rows;
            columns = gradient.columns;
            m.assign(gradient.data.size(), 0.0f);
            v.assign(gradient.data.size(), 0.0f);
        }
    };

    void update(std::vector<float>& gradient, State& state, float correction1, float correction2) {
        // Ein Durchlauf, keine temporaeren Matrizen oder Broadcast-Indexierung.
        for (size_t i = 0; i < gradient.size(); ++i) {
            const float g = gradient[i];
            state.m[i] = config.beta1 * state.m[i] + (1 - config.beta1) * g;
            state.v[i] = config.beta2 * state.v[i] + (1 - config.beta2) * g * g;
            gradient[i] = (state.m[i] * correction1)
                / (std::sqrt(state.v[i] * correction2) + config.epsilon);
        }
    }

    const Config config;
    State weights;
    State bias;
    size_t t = 0;
};
