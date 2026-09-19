// Implementierung der Layer
#include "layer.hpp"
#include "../matrix/matrix.hpp"
#include "../activation/activation.hpp"
#include <stdexcept>

void Layer::forward(const Matrix<Layout::ColumnMajor>& activation, const ActivationFunction& activation_function)
{

    if (W.columns != activation.rows)
        throw std::invalid_argument("Anzahl Gewichte und Koeffizienten stimmt nicht");

    a = W * activation;
    a += b;
   activation_function.forward(a);
}

Matrix<Layout::ColumnMajor> Layer::backward(Matrix<Layout::ColumnMajor>& dL, const ActivationFunction& activation_function, const Matrix<Layout::ColumnMajor>& a_prev, float learning_rate){
    activation_function.backward(dL, a); // = dL/dz

    b -=  dL.sum_across_columns() * (learning_rate * 1.0f / batch_size);
    dW = (dL * a_prev.T()) * (1.0f / batch_size);
    Matrix<Layout::ColumnMajor> result = W.T() * dL;
    W -= (dW * learning_rate);

    return result;
}
