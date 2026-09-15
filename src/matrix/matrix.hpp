// Matrix Klasse zum speichern von Gewichten, Zwischenergebnissen und Ableitungen
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

template<Layout L>
struct TransposeExpr;

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
        // Unitialazation = reserve, Zero = resize, Random = random()
        Matrix(size_t rows, size_t columns, Initialization typ);
        Matrix(std::vector<float>&& data, size_t rows, size_t columns);

        // ermöglicht Zugriff über ()
        float& operator()(size_t row, size_t column);
        float operator()(size_t row, size_t column) const;

        // vielleicht brauch ich es mal
        void resize(size_t row, size_t column);

        // Skalarmultiplikation
        void operator*(float number);

        // Multiplikation zweier Matrizen
        Matrix<Layout::ColumnMajor> operator*(const Matrix<Layout::ColumnMajor>& Other);

        // Ist nur ein View für '.T() *' transpose_multiply(A, B)
        TransposeExpr<L> T() const;

        // einfachrer Print
        void print() const;
};

template<Layout L>
struct TransposeExpr {
    const Matrix<L>& ref;

    inline size_t rows() const {
        return ref.columns;
    }

    inline size_t columns() const {
        return ref.rows;
    }

    inline float operator()(size_t row, size_t column) const {
        return ref(column, row);
    }
};

template<Layout L>
inline TransposeExpr<L> Matrix<L>::T() const {
    return TransposeExpr<L>{*this};
}

// Für RowMajor x ColumnMajor SIMD optimiert,
// funktioniert aber auch für ColumnMajor x ColumnMajor
template<Layout L>
Matrix<Layout::ColumnMajor> operator*(const TransposeExpr<L>& A, const Matrix<Layout::ColumnMajor>& B);

// Für ColumnMajor x ColumnMajor SIMD optimiert,
// funktioniert aber auch für ColumnMajor x RowMajor
template<Layout L>
Matrix<Layout::ColumnMajor> operator*(const Matrix<Layout::ColumnMajor>& A, const TransposeExpr<L>& B);
