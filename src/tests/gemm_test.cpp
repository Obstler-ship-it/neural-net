#include "../matrix/gemm.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

void check_product(size_t rows, size_t shared, size_t columns, bool transpose_left) {
    std::vector<float> left(rows * shared), right(shared * columns), result(rows * columns, 0.125f);
    for (size_t index = 0; index < left.size(); ++index)
        left[index] = 0.03f * (static_cast<int>(index % 19) - 9);
    for (size_t index = 0; index < right.size(); ++index)
        right[index] = 0.05f * (static_cast<int>(index % 23) - 11);
    matrix_detail::multiply_add(left.data(), right.data(), result.data(), rows, shared, columns,
                                transpose_left ? 1 : shared, transpose_left ? rows : 1);
    for (size_t row = 0; row < rows; ++row)
        for (size_t column = 0; column < columns; ++column) {
            double expected = 0.125;
            for (size_t index = 0; index < shared; ++index)
                expected += static_cast<double>(left[transpose_left ? index * rows + row : row * shared + index])
                          * right[index * columns + column];
            const float actual = result[row * columns + column];
            if (!std::isfinite(actual) || std::abs(actual - expected) > 0.00001 * (1 + std::abs(expected)))
                throw std::runtime_error("Blocked matrix multiplication differs from reference");
        }
}

int main() {
    for (size_t rows : {1, 3, 4, 5, 8, 17})
        for (size_t shared : {1, 3, 17, 33, 576})
            for (size_t columns : {1, 7, 15, 16, 17, 31, 32, 33, 64})
                for (bool transpose_left : {false, true})
                    check_product(rows, shared, columns, transpose_left);
    std::cout << "Alle blockierten Matrixprodukte erfolgreich.\n";
}
