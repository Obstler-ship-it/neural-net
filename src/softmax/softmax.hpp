#pragma once

#include "../matrix/matrix.hpp"
#include <cmath>
#include <cstddef>
#include <vector>
#include <stdexcept>

namespace softmax{

    // Forward für Softmax
    void forward(Matrix<Layout::ColumnMajor>& z);

    // Multiplikation nach der Kettenregel mit der entsprechenden Ableitung
    Matrix<Layout::ColumnMajor> backward(const Matrix<Layout::ColumnMajor>& dL, const Matrix<Layout::ColumnMajor>& y_hat);
};


inline void softmax::forward(Matrix<Layout::ColumnMajor>& z){

    for (size_t i=0; i < z.columns; i++){
        float max_value = z(0, i);

        for (std::size_t j=0; j < z.rows; j++) {
            max_value = std::max(max_value, z(j,i));
        }

        float sum = 0.0f;

        for(size_t j=0; j < z.rows; j++){
            z(j,i) = std::exp(z(j,i) - max_value);
            sum += z(j,i);
        }

        for(size_t j=0; j < z.rows; j++){
            z(j,i) = z(j,i) / sum;
        }
    }

}

inline Matrix<Layout::ColumnMajor> softmax::backward(
    const Matrix<Layout::ColumnMajor>& dL, const Matrix<Layout::ColumnMajor>& y_hat) {
    if (dL.rows != y_hat.rows || dL.columns != y_hat.columns)
        throw std::invalid_argument("Softmax: gradient and prediction dimensions differ");

    Matrix<Layout::ColumnMajor> dz(y_hat.rows, y_hat.columns, Initialization::Zero);
    for (size_t column = 0; column < y_hat.columns; ++column) {
        float dot = 0.0f;
        for (size_t row = 0; row < y_hat.rows; ++row)
            dot += dL(row, column) * y_hat(row, column);
        for (size_t row = 0; row < y_hat.rows; ++row)
            dz(row, column) = y_hat(row, column) * (dL(row, column) - dot);
    }
    return dz;
}
