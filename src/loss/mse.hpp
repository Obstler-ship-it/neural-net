#pragma once

#include "loss_function.hpp"
#include <cstddef>
#include <stdexcept>

class MeanSquareError : public LossFunction {
public:
    // Mittelwert ueber Klassen, Summe ueber Beispiele.
    float forward(const Matrix<Layout::ColumnMajor>& y_hat, const std::vector<int>& y) const override {
        validate(y_hat, y);
        float sum = 0.0f;
        for (size_t column = 0; column < y_hat.columns; ++column) {
            for (size_t row = 0; row < y_hat.rows; ++row) {
                const float target = row == static_cast<size_t>(y[column]) ? 1.0f : 0.0f;
                const float error = y_hat(row, column) - target;
                sum += error * error;
            }
        }
        return sum / static_cast<float>(y_hat.rows);
    }

    Matrix<Layout::ColumnMajor> backward(const Matrix<Layout::ColumnMajor>& y_hat,
                                       const std::vector<int>& y) const override {
        validate(y_hat, y);
        Matrix<Layout::ColumnMajor> gradient = y_hat;
        for (size_t column = 0; column < y_hat.columns; ++column)
            gradient(static_cast<size_t>(y[column]), column) -= 1.0f;
        gradient *= 2.0f / static_cast<float>(y_hat.rows);
        return gradient;
    }

    // Unnormalisierte Ableitung eines einzelnen Terms mit Zielwert 1.
    float derivative(float y_hat) const override {
        return 2.0f * (y_hat - 1.0f);
    }

private:
    static void validate(const Matrix<Layout::ColumnMajor>& y_hat, const std::vector<int>& y) {
        if (y_hat.rows == 0 || y_hat.columns != y.size())
            throw std::invalid_argument("MSE: invalid prediction dimensions or label count");
        for (int label : y)
            if (label < 0 || static_cast<size_t>(label) >= y_hat.rows)
                throw std::invalid_argument("MSE: label outside output range");
    }
};
