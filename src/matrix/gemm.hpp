#pragma once

#include <algorithm>
#include <cstddef>
#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#endif

namespace matrix_detail {
inline void multiply_add(const float* left, const float* right, float* result,
                         size_t rows, size_t shared, size_t columns,
                         size_t left_row_stride, size_t left_column_stride=1) {
    size_t row = 0;
#if defined(__AVX2__) && defined(__FMA__)
    for (; row + 4 <= rows; row += 4) {
        size_t column = 0;
        for (; column + 16 <= columns; column += 16) {
            auto sum00 = _mm256_loadu_ps(result + row * columns + column);
            auto sum01 = _mm256_loadu_ps(result + row * columns + column + 8);
            auto sum10 = _mm256_loadu_ps(result + (row + 1) * columns + column);
            auto sum11 = _mm256_loadu_ps(result + (row + 1) * columns + column + 8);
            auto sum20 = _mm256_loadu_ps(result + (row + 2) * columns + column);
            auto sum21 = _mm256_loadu_ps(result + (row + 2) * columns + column + 8);
            auto sum30 = _mm256_loadu_ps(result + (row + 3) * columns + column);
            auto sum31 = _mm256_loadu_ps(result + (row + 3) * columns + column + 8);
            for (size_t index = 0; index < shared; ++index) {
                const auto right0 = _mm256_loadu_ps(right + index * columns + column);
                const auto right1 = _mm256_loadu_ps(right + index * columns + column + 8);
                const float* values = left + row * left_row_stride + index * left_column_stride;
                const auto left0 = _mm256_set1_ps(values[0]);
                const auto left1 = _mm256_set1_ps(values[left_row_stride]);
                const auto left2 = _mm256_set1_ps(values[2 * left_row_stride]);
                const auto left3 = _mm256_set1_ps(values[3 * left_row_stride]);
                sum00 = _mm256_fmadd_ps(left0, right0, sum00);
                sum01 = _mm256_fmadd_ps(left0, right1, sum01);
                sum10 = _mm256_fmadd_ps(left1, right0, sum10);
                sum11 = _mm256_fmadd_ps(left1, right1, sum11);
                sum20 = _mm256_fmadd_ps(left2, right0, sum20);
                sum21 = _mm256_fmadd_ps(left2, right1, sum21);
                sum30 = _mm256_fmadd_ps(left3, right0, sum30);
                sum31 = _mm256_fmadd_ps(left3, right1, sum31);
            }
            _mm256_storeu_ps(result + row * columns + column, sum00);
            _mm256_storeu_ps(result + row * columns + column + 8, sum01);
            _mm256_storeu_ps(result + (row + 1) * columns + column, sum10);
            _mm256_storeu_ps(result + (row + 1) * columns + column + 8, sum11);
            _mm256_storeu_ps(result + (row + 2) * columns + column, sum20);
            _mm256_storeu_ps(result + (row + 2) * columns + column + 8, sum21);
            _mm256_storeu_ps(result + (row + 3) * columns + column, sum30);
            _mm256_storeu_ps(result + (row + 3) * columns + column + 8, sum31);
        }
        for (size_t tail_row = row; tail_row < row + 4; ++tail_row)
            for (size_t index = 0; index < shared; ++index) {
                const float value = left[tail_row * left_row_stride + index * left_column_stride];
                for (size_t tail_column = column; tail_column < columns; ++tail_column)
                    result[tail_row * columns + tail_column] += value * right[index * columns + tail_column];
            }
    }
#endif
    for (; row < rows; ++row)
        for (size_t index = 0; index < shared; ++index) {
            const float value = left[row * left_row_stride + index * left_column_stride];
            for (size_t column = 0; column < columns; ++column)
                result[row * columns + column] += value * right[index * columns + column];
        }
}
}
