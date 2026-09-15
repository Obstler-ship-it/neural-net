#pragma once

#include "loss_function.hpp"
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

class CrossEntropy: public LossFunction{
    public:

        float forward(const Matrix<Layout::ColumnMajor>& y_hat, const std::vector<int>& y) const override{

            if (y_hat.columns != y.size())
                throw std::invalid_argument("Batch und Anzahl Labels haben unterschiedliche Größe!");

            float loss = 0;

            for (size_t i=0; i< y_hat.columns; i++){
                loss += -std::log(y_hat(y[i],i));
            }

            return loss;
        }

        float derivative(float y_hat) const override{
            float dt = - 1.0f / y_hat;
            return dt;
        }

};
