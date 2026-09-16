#pragma once

#include "activation.hpp"
#include "../matrix/matrix.hpp"
#include <cmath>
#include <vector>

class Sigmoid: public ActivationFunction {
    public:
        Matrix<Layout::ColumnMajor> foward(const Matrix<Layout::ColumnMajor>& z) const override {
            Matrix<Layout::ColumnMajor> result(z.rows, z.columns, Initialization::Uninitialized);

            for (float number: z.data) {
                result.data.push_back(1.0f/(1.0f + std::exp(-number)));
            }

            return result;
        }

        Matrix<Layout::ColumnMajor> backward(const Matrix<Layout::ColumnMajor>& z) const override{
            Matrix<Layout::ColumnMajor> result(z.rows, z.columns, Initialization::Uninitialized);

            for (float number: z.data) {
                result.data.push_back(number * (1 - number));
            }

            return result;
        }
};
