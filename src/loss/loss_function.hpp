// interface der loss_functions des NNs
#pragma once

#include "../matrix/matrix.hpp"
#include <stdexcept>

class LossFunction{
    public:
        virtual float forward(const Matrix<Layout::ColumnMajor>& z, const std::vector<int>& y) const = 0;

        inline virtual float derivative(float) const = 0;

        virtual Matrix<Layout::ColumnMajor> backward(const Matrix<Layout::ColumnMajor>& y_hat, const std::vector<int>& y) {

            if (y_hat.columns != y.size())
                throw std::invalid_argument("Batch und Anzahl Labels haben unterschiedliche Größe!");

            Matrix<Layout::ColumnMajor> dL(y_hat.rows, y_hat.columns, Initialization::Uninitialized);

            for (size_t i=0; i < y_hat.columns; i++){
                float dt = derivative(y_hat(y[i], i));
                for (size_t j=0; j < y_hat.rows; j++){
                    dL.data.push_back(dt);
                }
            }

            return dL;
        }

        virtual ~LossFunction() = default;
};
