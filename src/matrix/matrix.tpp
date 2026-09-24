#pragma once

#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>

inline std::mt19937& random_generator() {
    static std::mt19937 generator{42};
    return generator;
}

inline float random_float() {
    static std::uniform_real_distribution<float> distribution{-1.0f, 1.0f};
    return distribution(random_generator());
}

template<Layout L>
Matrix<L>::Matrix(size_t rows, size_t columns, Initialization typ)
    : rows(rows), columns(columns) {
    if (typ == Initialization::Zero) {
        data.resize(rows * columns);
    } else if (typ == Initialization::Uninitialized) {
        data.reserve(rows * columns);
    } else if (typ == Initialization::Random) {
        data.reserve(rows * columns);
        for (size_t index = 0; index < rows * columns; ++index) {
            data.push_back(random_float());
        }
    }
}

template<Layout L>
Matrix<L>::Matrix(std::vector<float>&& data, size_t rows, size_t columns)
    : data(std::move(data)), rows(rows), columns(columns) {}

template<>
inline float Matrix<Layout::ColumnMajor>::operator()(size_t row, size_t column) const {
    return data[rows * column + row];
}

template<>
inline float& Matrix<Layout::ColumnMajor>::operator()(size_t row, size_t column) {
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
void Matrix<L>::resize(size_t row, size_t column) {
    rows = row;
    columns = column;
    data.resize(rows * columns);
}

template<Layout L>
Matrix<L>& Matrix<L>::operator*(float number) {
    for (auto& value : data) {
        value *= number;
    }
    return *this;
}

template<>
inline Matrix<Layout::ColumnMajor> Matrix<Layout::RowMajor>::operator*(const Matrix<Layout::ColumnMajor>& other) {
    if (columns != other.rows) {
        throw std::invalid_argument("Matrizen haben inkompatible Dimensionen");
    }

    Matrix<Layout::ColumnMajor> result(rows, other.columns, Initialization::Zero);
    for (size_t row = 0; row < rows; ++row) {
        for (size_t column = 0; column < other.columns; ++column) {
            for (size_t index = 0; index < columns; ++index) {
                result(row, column) += (*this)(row, index) * other(index, column);
            }
        }
    }
    return result;
}

template<Layout L>
Matrix<Layout::ColumnMajor> operator*(const TransposeExpr<L>& first, const Matrix<Layout::ColumnMajor>& second) {
    if (first.columns() != second.rows) {
        throw std::invalid_argument("Matrizen haben inkompatible Dimensionen");
    }

    Matrix<Layout::ColumnMajor> result(first.rows(), second.columns, Initialization::Zero);
    for (size_t column = 0; column < second.columns; ++column) {
        for (size_t shared = 0; shared < second.rows; ++shared) {
            for (size_t row = 0; row < first.rows(); ++row) {
                result(row, column) += first(row, shared) * second(shared, column);
            }
        }
    }
    return result;
}

template<Layout L>
Matrix<Layout::ColumnMajor> operator*(const Matrix<Layout::ColumnMajor>& first, const TransposeExpr<L>& second) {
    if (first.columns != second.rows()) {
        throw std::invalid_argument("Matrizen haben inkompatible Dimensionen");
    }

    Matrix<Layout::ColumnMajor> result(first.rows, second.columns(), Initialization::Zero);
    for (size_t column = 0; column < second.columns(); ++column) {
        for (size_t shared = 0; shared < first.columns; ++shared) {
            for (size_t row = 0; row < first.rows; ++row) {
                result(row, column) += first(row, shared) * second(shared, column);
            }
        }
    }
    return result;
}

template<>
template<Layout OtherLayout>
inline Matrix<Layout::ColumnMajor>& Matrix<Layout::ColumnMajor>::operator+=(const Matrix<OtherLayout>& other) {
    if (other.data.empty()) {
        throw std::invalid_argument("Matrix ist leer");
    }
    for (size_t column = 0; column < columns; ++column) {
        for (size_t row = 0; row < rows; ++row) {
            (*this)(row, column) += other(row % other.rows, column % other.columns);
        }
    }
    return *this;
}

template<>
template<Layout OtherLayout>
inline Matrix<Layout::RowMajor>& Matrix<Layout::RowMajor>::operator+=(const Matrix<OtherLayout>& other) {
    if (other.data.empty()) {
        throw std::invalid_argument("Matrix ist leer");
    }
    for (size_t row = 0; row < rows; ++row) {
        for (size_t column = 0; column < columns; ++column) {
            (*this)(row, column) += other(row % other.rows, column % other.columns);
        }
    }
    return *this;
}

template<>
template<Layout OtherLayout>
inline Matrix<Layout::ColumnMajor>& Matrix<Layout::ColumnMajor>::operator-=(const Matrix<OtherLayout>& other) {
    if (other.data.empty()) {
        throw std::invalid_argument("Matrix ist leer");
    }
    for (size_t column = 0; column < columns; ++column) {
        for (size_t row = 0; row < rows; ++row) {
            (*this)(row, column) -= other(row % other.rows, column % other.columns);
        }
    }
    return *this;
}

template<>
template<Layout OtherLayout>
inline Matrix<Layout::RowMajor>& Matrix<Layout::RowMajor>::operator-=(const Matrix<OtherLayout>& other) {
    if (other.data.empty()) {
        throw std::invalid_argument("Matrix ist leer");
    }
    for (size_t row = 0; row < rows; ++row) {
        for (size_t column = 0; column < columns; ++column) {
            (*this)(row, column) -= other(row % other.rows, column % other.columns);
        }
    }
    return *this;
}

template<Layout L>
Matrix<L> Matrix<L>::sum_across_columns() {
    if (data.empty()) {
        throw std::logic_error("Matrix ist leer!!");
    }

    Matrix<L> result(rows, 1, Initialization::Uninitialized);
    for (size_t row = 0; row < rows; ++row) {
        float sum = 0.0f;
        for (size_t column = 0; column < columns; ++column) {
            sum += (*this)(row, column);
        }
        result.data.push_back(sum);
    }
    return result;
}

template<Layout L>
void Matrix<L>::print() const {
    std::cout << "[\n";
    for (size_t row = 0; row < rows; ++row) {
        std::cout << "  [ ";
        for (size_t column = 0; column < columns; ++column) {
            std::cout << std::setw(9) << std::fixed << std::setprecision(4) << (*this)(row, column);
        }
        std::cout << " ]\n";
    }
    std::cout << "]\n";
}
