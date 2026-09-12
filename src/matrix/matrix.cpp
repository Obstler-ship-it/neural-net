#include "matrix.hpp"

#include <cstddef>
#include <sys/types.h>
#include <stdexcept>
#include <vector>
#include <random>


inline std::mt19937& random_generator() {
    static std::mt19937 generator{42}; // per Hand gesetzt nicht vergessen!!
    return generator;
}

inline float random_float() {
    static std::uniform_real_distribution<float> distribution{-1.0f, 1.0f};
    return distribution(random_generator());
}

template<Layout L>
struct TransposeExpr{
    const Matrix<L>& ref;

    size_t rows() const {
        return ref.columns;
    }
};

template<Layout L>
TransposeExpr<L> Matrix<L>::T() const {
    return TransposeExpr{*this};
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
Matrix<L>::Matrix(const std::vector<float>&& data, size_t rows, size_t columns)
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
void Matrix<L>::operator*(float number){
    for (auto& x: data)
        x *= number;
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

    for (size_t i=0; i < B.columns; i++){
        for (size_t j=0; j < B.rows; j++){
            for (size_t k=0; k < A.rows(); k++){
                C(k,i) += A.ref(j,k) * B(j,i);
            }
        }

    }
    return C;
}
