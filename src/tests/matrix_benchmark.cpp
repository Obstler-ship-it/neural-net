// Komplett von Codex geschrieben
#include "../matrix/matrix.hpp"

#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string_view>

namespace {

constexpr size_t input_size = 784;
constexpr size_t hidden_size = 64;
constexpr size_t output_size = 96;

template<typename Operation>
void benchmark(std::string_view name, size_t iterations, Operation operation) {
    float checksum = 0.0f;
    const auto start = std::chrono::steady_clock::now();

    for (size_t iteration = 0; iteration < iterations; ++iteration) {
        const auto result = operation();
        checksum += result(0, 0) + result(result.rows - 1, result.columns - 1);
    }

    const auto elapsed = std::chrono::steady_clock::now() - start;
    const double milliseconds = std::chrono::duration<double, std::milli>(elapsed).count();

    std::cout << std::left << std::setw(28) << name
              << std::right << std::fixed << std::setprecision(3)
              << std::setw(10) << milliseconds << " ms total  "
              << std::setw(8) << milliseconds / static_cast<double>(iterations)
              << " ms/op  checksum=" << checksum << '\n';
}

}

int main(int argc, char* argv[]) {
    const size_t iterations = argc == 2
        ? static_cast<size_t>(std::strtoull(argv[1], nullptr, 10))
        : 50;

    if (iterations == 0) {
        std::cerr << "Die Anzahl der Iterationen muss groesser als null sein.\n";
        return 1;
    }

    Matrix<Layout::RowMajor> normal_left(input_size, hidden_size, Initialization::Random);
    Matrix<Layout::ColumnMajor> normal_right(hidden_size, output_size, Initialization::Random);

    Matrix<Layout::RowMajor> transpose_left_source(hidden_size, input_size, Initialization::Random);
    Matrix<Layout::ColumnMajor> transpose_left_right(hidden_size, output_size, Initialization::Random);

    Matrix<Layout::ColumnMajor> transpose_right_left(input_size, hidden_size, Initialization::Random);
    Matrix<Layout::ColumnMajor> transpose_right_source(output_size, hidden_size, Initialization::Random);

    std::cout << "Matrix benchmark: " << input_size << 'x' << hidden_size
              << " * " << hidden_size << 'x' << output_size
              << ", " << iterations << " Iterationen\n";

    benchmark("RowMajor * ColumnMajor", iterations, [&] {
        return normal_left * normal_right;
    });

    benchmark("Transpose(RowMajor) * ColumnMajor", iterations, [&] {
        return transpose_left_source.T() * transpose_left_right;
    });

    benchmark("ColumnMajor * Transpose(ColumnMajor)", iterations, [&] {
        return transpose_right_left * transpose_right_source.T();
    });
}

// -g -O3 ffast-math
// RowMajor * ColumnMajor                 0.376 ms/op
// Transpose(RowMajor) * ColumnMajor      0.436 ms/op
// ColumnMajor * Transpose(ColumnMajor)   0.298 ms/op

// -O0
// RowMajor * ColumnMajor                 57.683 ms/op
// Transpose(RowMajor) * ColumnMajor      67.472 ms/op
// ColumnMajor * Transpose(ColumnMajor)   41.896 ms/op
