#pragma once

#include "activation.hpp"
#include "matrix/matrix.hpp"
#include <vector>

class ReLu: public ActivationFunction {
    public:
        Matrix<Layout::ColumnMajor> activate(const Matrix<Layout::ColumnMajor>& z) const override{
            Matrix<Layout::ColumnMajor> result(z.rows, z.columns, Initialization::Uninitialized);

            for (float number: z.data) {
                result.data.push_back(std::max(0.0f, number));
            }

            return result;
        }

        Matrix<Layout::ColumnMajor> derivative(const Matrix<Layout::ColumnMajor>& z) const override{
            Matrix<Layout::ColumnMajor> result(z.rows, z.columns, Initialization::Uninitialized);

            for (float number: z.data) {
                if (number > 0)
                    result.data.push_back(1);
                else
                    result.data.push_back(0);
            }

            return result;
        }
};
