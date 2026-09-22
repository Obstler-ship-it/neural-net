#pragma once

#include "activation.hpp"
#include "../matrix/matrix.hpp"
#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <vector>

class ReLU: public ActivationFunction {
    public:
        void forward(Matrix<Layout::ColumnMajor>& z) const override{

            for (float& number: z.data) {
                number = std::max(0.0f, number);
            }

        }

        void backward(Matrix<Layout::ColumnMajor>& dL, const Matrix<Layout::ColumnMajor>& a) const override{

            if (dL.rows != a.rows || dL.columns != a.columns)
                throw std::invalid_argument("Matrizen Dimensionen stimmen nicht überein!");

            for (size_t i=0; i < a.data.size(); i++){
                if (a.data[i] <= 0.0f)
                    dL.data[i] = 0.0f;
            }

        }
};
