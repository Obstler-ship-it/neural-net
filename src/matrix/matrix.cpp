#include "matrix.hpp"

#include <cstddef>
#include <sys/types.h>
#include <stdexcept>
#include <vector>

template<>
inline float Matrix<Layout::ColumnMajor>::operator()(size_t row, size_t column) const{
    return data[rows * column + row];
}

template<>
inline float& Matrix<Layout::ColumnMajor>::operator()(size_t row, size_t column){
    return data[rows * column + row];
}

template<>
inline float Matrix<Layout::RowMajor>::operator()(size_t row, size_t column) const {
    return data[columns * row + column];
}

template<>
inline float& Matrix<Layout::RowMajor>::operator()(size_t row, size_t column) {
    return data[columns * row + column];
}

template<Layout L>
Matrix<L>::Matrix(size_t rows, size_t columns, bool zero)
    : rows(rows), columns(columns) {
        if (zero)
            data.resize(rows * columns);
        else
            data.reserve(rows * columns);
    }

template<Layout L>
void Matrix<L>::resize(size_t row, size_t column){
    rows = row;
    columns = column;
    data.resize(rows * columns);
}

template<Layout L>
void Matrix<L>::operator*(float number){
    for (auto& x: data)
        x *= number;
}

template<>
Matrix<Layout::ColumnMajor> Matrix<Layout::RowMajor>::operator*(const Matrix<Layout::ColumnMajor>& other){

    if (columns != other.rows)
        throw std::invalid_argument("Matrizen haben inkompatible Dimensionen");

    Matrix<Layout::ColumnMajor> result(this->rows, other.columns, true);

    for (size_t i=0; i < rows; i++) {
        for (size_t j=0; j < other.columns; j++) {
            for (size_t k=0; k < columns; k++) {
                result(i , j) += (*this)(i, k) * other(k, j);
            }
        }
    }

    return result;
}
