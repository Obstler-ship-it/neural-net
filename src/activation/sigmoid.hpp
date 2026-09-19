#pragma once

#include "activation.hpp"
#include "../matrix/matrix.hpp"
#include <cmath>
#include <stdexcept>
#include <vector>

class Sigmoid: public ActivationFunction {
    public:
        void forward(Matrix<Layout::ColumnMajor>& z) const override {

            for (float& number: z.data) {
                number = (1.0f/(1.0f + std::exp(-number)));
            }

        }

        void backward(Matrix<Layout::ColumnMajor>& dL, const Matrix<Layout::ColumnMajor>& a) const override{

            if (dL.rows != a.rows || dL.columns != a.columns)
                throw std::invalid_argument("Matrizen Dimensionen stimmen nicht überein!");

            for (size_t i=0; i < a.data.size(); i++){
                dL.data[i] *= a.data[i] * (1.0f - a.data[i]);
            }
        }
};
