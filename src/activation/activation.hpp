#pragma once

#include "../matrix/matrix.hpp"

/*
 * Nichtlineare Aktivierungsfunktion
 */
class ActivationFunction {
public:
    virtual Matrix<Layout::ColumnMajor> activate(const Matrix<Layout::ColumnMajor>& z) const = 0;
    virtual Matrix<Layout::ColumnMajor> derivative(const Matrix<Layout::ColumnMajor>& z) const = 0;
    virtual ~ActivationFunction() = default;
};
