#include "matrix.hpp"

#include <cstddef>
#include <sys/types.h>
#include <stdexcept>
#include <vector>
#include <random>
#include <iostream>
#include <iomanip>


inline std::mt19937& random_generator() {
    static std::mt19937 generator{42}; // per Hand gesetzt nicht vergessen!!
    return generator;
}

inline float random_float() {
    static std::uniform_real_distribution<float> distribution{-1.0f, 1.0f};
    return distribution(random_generator());
}

template<Layout L>
Matrix<L>::Matrix(size_t rows, size_t columns, Initialization typ)
    : rows(rows), columns(columns) {
        if (typ == Initialization::Zero)
            data.resize(rows * columns);
        else if (typ == Initialization::Uninitialized)
            data.reserve(rows * columns);
        else if (typ == Initialization::Random) {
            data.reserve(rows * columns);

            for (size_t i = 0; i < rows * columns; ++i) {
                data.push_back(random_float());
            }
        }

    }

template<Layout L>
Matrix<L>::Matrix(std::vector<float>&& data, size_t rows, size_t columns)
    : data(std::move(data)), rows(rows), columns(columns) {}

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
void Matrix<L>::resize(size_t row, size_t column){
    rows = row;
    columns = column;
    data.resize(rows * columns);
}

template<Layout L>
Matrix<L>& Matrix<L>::operator*(float number){
    for (auto& x: data)
        x *= number;
    return *this;
}

template<>
Matrix<Layout::ColumnMajor> Matrix<Layout::RowMajor>::operator*(const Matrix<Layout::ColumnMajor>& other){

    if (columns != other.rows)
        throw std::invalid_argument("Matrizen haben inkompatible Dimensionen");

    Matrix<Layout::ColumnMajor> result(this->rows, other.columns, Initialization::Zero);

    for (size_t i=0; i < rows; i++) {
        for (size_t j=0; j < other.columns; j++) {
            for (size_t k=0; k < columns; k++) {
                result(i , j) += (*this)(i, k) * other(k, j);
            }
        }
    }

    return result;
}

template<Layout L>
Matrix<Layout::ColumnMajor> operator*(const TransposeExpr<L>& A, const Matrix<Layout::ColumnMajor>& B){
    Matrix<Layout::ColumnMajor> C (A.rows(), B.columns, Initialization::Zero);

    if (A.columns() != B.rows)
        throw std::invalid_argument("Matrizen haben inkompatible Dimensionen");

    for (size_t i=0; i < B.columns; i++){
        for (size_t j=0; j < B.rows; j++){
            for (size_t k=0; k < A.rows(); k++){
                C(k,i) += A(k,j) * B(j,i);
            }
        }
    }

    return C;
}

template<Layout L>
Matrix<Layout::ColumnMajor> operator*(const Matrix<Layout::ColumnMajor>& A, const TransposeExpr<L>& B){
    Matrix<Layout::ColumnMajor> C (A.rows, B.columns(), Initialization::Zero);

    if (A.columns != B.rows())
        throw std::invalid_argument("Matrizen haben inkompatible Dimensionen");

    for (size_t i=0; i < B.rows(); i++){
        for (size_t j=0; j < A.columns; j++){
            for (size_t k=0; k < A.rows; k++){
                C(k,i) += A(k,j) * B(j,i);
            }
        }
    }

    return C;
}

template<>
template<Layout OtherLayout>
Matrix<Layout::ColumnMajor>& Matrix<Layout::ColumnMajor>::operator+=(const Matrix<OtherLayout>& other){

    if (other.data.size() < 1)
        throw std::invalid_argument("Matrix ist leer");

    for (size_t i=0; i < this->columns; i++){
        for (size_t j=0; j < this->rows; j++){
            (*this)(j,i) += other(j % other.rows,i % other.columns);
        }
    }

    return *this;
}

template<>
template<Layout OtherLayout>
Matrix<Layout::RowMajor>& Matrix<Layout::RowMajor>::operator+=(const Matrix<OtherLayout>& other){

    if (other.data.size() < 1)
        throw std::invalid_argument("Matrix ist leer");

    for (size_t i=0; i < this->rows; i++){
        for (size_t j=0; j < this->columns; j++){
            (*this)(i,j) += other(i % other.rows,j % other.columns);
        }
    }

    return *this;
}

template<>
template<Layout OtherLayout>
Matrix<Layout::ColumnMajor>& Matrix<Layout::ColumnMajor>::operator-=(const Matrix<OtherLayout>& other){

    if (other.data.size() < 1)
        throw std::invalid_argument("Matrix ist leer");

    for (size_t i=0; i < this->columns; i++){
        for (size_t j=0; j < this->rows; j++){
            (*this)(j,i) -= other(j % other.rows,i % other.columns);
        }
    }

    return *this;
}

template<>
template<Layout OtherLayout>
Matrix<Layout::RowMajor>& Matrix<Layout::RowMajor>::operator-=(const Matrix<OtherLayout>& other){

    if (other.data.size() < 1)
        throw std::invalid_argument("Matrix ist leer");

    for (size_t i=0; i < this->rows; i++){
        for (size_t j=0; j < this->columns; j++){
            (*this)(i,j) -= other(i % other.rows,j % other.columns);
        }
    }

    return *this;
}

template<Layout L>
Matrix<L> Matrix<L>::sum_across_columns(){

    if (data.size() < 1)
        throw std::logic_error("Matrix ist leer!!");

    Matrix<L> result(this->rows, 1, Initialization::Uninitialized);
    for (size_t i=0; i < this->rows; i++){
        float sum = 0;
        for (size_t j=0; j < this->columns; j++){
            sum += (*this)(i,j);
        }
        result.data.push_back(sum);
    }

    return result;
}

template<Layout L>
void Matrix<L>::print() const {
    std::cout << "[\n";

    for (size_t i = 0; i < rows; ++i) {
        std::cout << "  [ ";

        for (size_t j = 0; j < columns; ++j) {
            std::cout << std::setw(9)
                      << std::fixed
                      << std::setprecision(4)
                      << (*this)(i, j);
        }

        std::cout << " ]\n";
    }

    std::cout << "]\n";
}
