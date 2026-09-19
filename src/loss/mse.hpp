#pragma once

#include "loss_function.hpp"
#include "../matrix/matrix.hpp"
#include <cmath>
#include <cstddef>

class MeanSquareError:public LossFunction{
    public:
        float forward(const Matrix<Layout::ColumnMajor>& y_hat, const std::vector<int>& y) const override{
            float sum = 0;

            for (size_t i=0; i < y_hat.columns; i++){
                sum += 1.0f/2.0f * std::pow((y_hat(y[i],i) - 1), 2);
            }

            return sum;
        }

        float derivative(float y_hat) const override{
            return (y_hat - 1);
        }
};
