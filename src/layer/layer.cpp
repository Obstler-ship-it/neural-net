#include "layer.hpp"
#include <stdexcept>

Layer::Layer(std::size_t rows, std::size_t columns)
    :rows(rows), columns(columns) 
    {
        layer.reserve(rows * columns);
    }

float& Layer::operator() (int row, int column){
    return layer[row * columns + column];
}

std::vector<float> Layer::dot(const std::vector<float> vector){

    if (vector.size() != columns)
        throw std::logic_error("Matrix Multiplikation fehlgeschlagen!");

    std::vector<float> result;
    result.resize(rows * vector.size());

    for(std::size_t i=0; i < rows; ++i){
        for(std::size_t j=0; j < columns; ++j){
            result[i] += (*this)(i, j) * vector[j];
        }
    }
    return result;
}