// interface der loss_functions des NNs
#pragma once

#include "../matrix/matrix.hpp"

class LossFunction{
    public:
        virtual Matrix<Layout::ColumnMajor> loss(const Matrix<Layout::ColumnMajor>& z) const = 0;
        virtual Matrix<Layout::ColumnMajor> derivative(const Matrix<Layout::ColumnMajor>& z) const = 0;
        virtual ~LossFunction() = default;
}
