#pragma once

#include "../matrix/matrix.hpp"
#include <cmath>
#include <cstddef>
#include <vector>

namespace softmax{

    // Forward für Softmax
    void forward(Matrix<Layout::ColumnMajor>& z);

    // Multiplikation nach der Kettenregel mit der entsprechenden Ableitung
    Matrix<Layout::ColumnMajor> backward(Matrix<Layout::ColumnMajor>& dL, const std::vector<int>& y, const Matrix<Layout::ColumnMajor>& y_hat);
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

inline Matrix<Layout::ColumnMajor> softmax::backward(Matrix<Layout::ColumnMajor>& dL, const std::vector<int>& y, const Matrix<Layout::ColumnMajor>& y_hat){

    Matrix<Layout::ColumnMajor> dz(y_hat.rows, y_hat.columns, Initialization::Zero);

    for (size_t i=0; i < dz.columns; i++){
        for (size_t j=0; j < dz.rows; j++){
            dz(j,i) = dL(0,i) * -y_hat(j,i) * y_hat(y[i],i);
        }
        dz(y[i], i) = dL(0, i) * y_hat(y[i],i) * (1-y_hat(y[i],i));
    }

    return dz;
}
