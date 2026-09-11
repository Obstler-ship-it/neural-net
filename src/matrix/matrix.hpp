#pragma once

#include <cstddef>
#include <sys/types.h>
#include <vector>

enum class Initialization {
    Zero,
    Uninitialized,
    Random
};

enum class Layout {
    RowMajor,
    ColumnMajor
};

/** Selbstgebauter Matrix Typ
 *
 * speichert die Elemente in einem std::vector<float>
 * kann Multiplikationen mit '*'
 * , etweder RowMajor oder ColumnMajor
 */
template<Layout L>
class Matrix {

    public:
        std::vector<float> data;
        size_t rows;
        size_t columns;

        // benötigt Reihen und Spalten Anzahl
        // zero = true macht resize sonst reserve
        Matrix(size_t rows, size_t columns, Initialization typ);

        // ermöglicht Zugriff über ()
        inline float& operator()(size_t row, size_t column);
        inline float operator()(size_t row, size_t column) const;

        // vielleicht brauch ich es mal
        void resize(size_t row, size_t column);

        // Skalarmultiplikation
        void operator*(float number);

        // Multiplikation zweier Matrizen
        Matrix<Layout::ColumnMajor> operator*(const Matrix<Layout::ColumnMajor>& Other);
};
