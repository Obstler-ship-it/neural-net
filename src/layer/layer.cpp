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
Matrix<Layout::ColumnMajor> Layer<Optimizer>::backward(Matrix<Layout::ColumnMajor>& dL, const ActivationFunction& activation_function, const Matrix<Layout::ColumnMajor>& a_prev, float learning_rate){
    activation_function.backward(dL, a); // = dL/dz

    db =  dL.sum_across_columns() * (learning_rate * 1.0f / batch_size);
    b -= db;
    dW = (dL * a_prev.T()) * (1.0f / batch_size);
    Matrix<Layout::ColumnMajor> result = W.T() * dL;
    W -= (dW * learning_rate);

    return result;
}

template<class Optimizer>
Matrix<Layout::ColumnMajor> Layer<Optimizer>::backward(Matrix<Layout::ColumnMajor>& dL, const Matrix<Layout::ColumnMajor>& a_prev, float learning_rate){

    db =  dL.sum_across_columns() * (learning_rate * 1.0f / batch_size);
    b -= db;
    dW = (dL * a_prev.T()) * (1.0f / batch_size);
    Matrix<Layout::ColumnMajor> result = W.T() * dL;
    W -= (dW * learning_rate);

    return result;
}

template<class Optimizer>
void Layer<Optimizer>::backward(float learning_rate, Matrix<Layout::ColumnMajor>& dL, const ActivationFunction& activation_function, const Matrix<Layout::ColumnMajor>& a_prev){
    activation_function.backward(dL, a); // = dL/dz

    db =  dL.sum_across_columns() * (learning_rate * 1.0f / batch_size);
    b -= db;
    dW = (dL * a_prev.T()) * (1.0f / batch_size);
    W -= (dW * learning_rate);
}
