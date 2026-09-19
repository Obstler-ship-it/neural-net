// interface der nicht linearen activation functions
#pragma once

#include "../matrix/matrix.hpp"

/*
 * Nichtlineare Aktivierungsfunktion
 */
class ActivationFunction {
public:
    virtual void forward(Matrix<Layout::ColumnMajor>& z) const = 0;
    virtual void backward(Matrix<Layout::ColumnMajor>& dL, const Matrix<Layout::ColumnMajor>& a) const = 0;
    virtual ~ActivationFunction() = default;
};
