// interface der nicht linearen activation functions
#pragma once

#include "../matrix/matrix.hpp"
#include <cstddef>

/*
 * Nichtlineare Aktivierungsfunktion
 */
class ActivationFunction {
public:
    //forward für die Aktivierungsfunktion
    virtual void forward(Matrix<Layout::ColumnMajor>& z) const = 0;
    //backward für die Aktivierungsfunktion
    virtual void backward(Matrix<Layout::ColumnMajor>& dL, const Matrix<Layout::ColumnMajor>& a) const = 0;
    //Skalierung für die Startwerte in den Matrizen, die sonst U~[-1,1] sind
    virtual float scale(size_t in, size_t out) const = 0;
    virtual ~ActivationFunction() = default;
};
