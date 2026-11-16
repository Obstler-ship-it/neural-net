#pragma once

// Implementierung der Layer
#include "layer.hpp"
#include "../matrix/matrix.hpp"
#include "../activation/activation.hpp"
#include <stdexcept>

template<class Optimizer>
void Layer<Optimizer>::forward(const Matrix<Layout::ColumnMajor>& activation, const ActivationFunction& activation_function)
{

    if (W.columns != activation.rows)
        throw std::invalid_argument("Anzahl Gewichte und Koeffizienten stimmt nicht");

    a = W * activation;
    a += b;
   activation_function.forward(a);
}

template<class Optimizer>
void Layer<Optimizer>::forward(const Matrix<Layout::ColumnMajor>& activation){

    if (W.columns != activation.rows)
        throw std::invalid_argument("Anzahl Gewichte und Koeffizienten stimmt nicht");

    a = W * activation;
    a += b;
}

template<class Optimizer>
Matrix<Layout::ColumnMajor> Layer<Optimizer>::backward(Matrix<Layout::ColumnMajor>& dL, const ActivationFunction& activation_function, const Matrix<Layout::ColumnMajor>& a_prev){
    activation_function.backward(dL, a); // = dL/dz

    db =  dL.sumAcrossColumns() *= (1.0f / batch_size);
    dW = (dL * a_prev.T()) *= (1.0f / batch_size);
    Matrix<Layout::ColumnMajor> result = W.T() * dL;
    optimizer.step(dW, db);
    b -= db;
    W -= dW;

    return result;
}

template<class Optimizer>
Matrix<Layout::ColumnMajor> Layer<Optimizer>::backward(Matrix<Layout::ColumnMajor>& dL, const Matrix<Layout::ColumnMajor>& a_prev){

    db =  dL.sumAcrossColumns() *= (1.0f / batch_size);
    dW = (dL * a_prev.T()) *= (1.0f / batch_size);
    Matrix<Layout::ColumnMajor> result = W.T() * dL;
    optimizer.step(dW, db);
    b -= db;
    W -= dW;

    return result;
}

template<class Optimizer>
void Layer<Optimizer>::backward(Matrix<Layout::ColumnMajor>& dL, const Matrix<Layout::ColumnMajor>& a_prev, const ActivationFunction& activation_function){
    activation_function.backward(dL, a); // = dL/dz

    db = dL.sumAcrossColumns() *= (1.0f / batch_size);
    dW = (dL * a_prev.T()) *= (1.0f / batch_size);
    optimizer.step(dW, db);
    b -= db;
    W -= dW;
}
