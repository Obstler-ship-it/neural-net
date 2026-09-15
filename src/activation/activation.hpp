// interface der nicht linearen activation functions
#pragma once

#include "../matrix/matrix.hpp"

/*
 * Nichtlineare Aktivierungsfunktion
 */
class ActivationFunction {
public:
    virtual Matrix<Layout::ColumnMajor> foward(const Matrix<Layout::ColumnMajor>& z) const = 0;
    virtual Matrix<Layout::ColumnMajor> backward(const Matrix<Layout::ColumnMajor>& z) const = 0;
    virtual ~ActivationFunction() = default;
};
