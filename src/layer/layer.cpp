// Implementierung der Layer
#include "layer.hpp"
#include "../matrix/matrix.hpp"
#include <stdexcept>

void Layer::forward(const Matrix<Layout::ColumnMajor>& activation){

    if (W.columns != activation.rows)
        throw std::invalid_argument("Anzahl Gewichte und Koeffizienten stimmt nicht");

    a = W * activation;
    a += b;
}

void Layer::backward()
