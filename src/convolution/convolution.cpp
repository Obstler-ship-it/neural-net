#include "convolution.hpp"
#include "matrix/matrix.hpp"
#include "mec/mec.hpp"
#include <cstddef>
#include <vector>

Matrix<Layout::ColumnMajor> Convolution::forward(const Images& input, std::span<const size_t> batch_indizes, const ActivationFunction& activation_function){
    const size_t output_rows = input.rows_per_image - KERNEL_SIZE + 1;
    const size_t output_columns = input.columns_per_image - KERNEL_SIZE + 1;
    const size_t output_size = output_rows * output_columns * NUMBER_KERNELS;

    Matrix<Layout::ColumnMajor> result(output_size, batch_indizes.size(), Initialization::Uninitialized);

    // Anzahl gespeicherter MEC-Werte pro Bild
    const size_t image_size = input.data.size() / input.number_images;
    const size_t strip_size = input.columns_per_image * KERNEL_SIZE;

    std::vector<float> sums(NUMBER_KERNELS);

    for (size_t image_index: batch_indizes) {
        const size_t image_offset = image_index * image_size;

        for (size_t y = 0; y < output_rows; ++y) {
            for (size_t x = 0; x < output_columns; ++x) {
                const size_t patch_offset = image_offset + y * strip_size + x * KERNEL_SIZE;

                for (size_t kernel = 0; kernel < NUMBER_KERNELS; ++kernel)
                    sums[kernel] = Bias.data[kernel];

                for (size_t idx = 0; idx < KERNEL_SIZE_SQUARED; ++idx) {
                    const float pixel = input.data[patch_offset + idx];

                    for (size_t kernel = 0; kernel < NUMBER_KERNELS; ++kernel)
                        sums[kernel] += pixel * Kernels(idx, kernel);
                }

                result.data.insert(result.data.end(), sums.begin(), sums.end());
            }
        }
    }

    activation_function.forward(result);
    return result;
}

Matrix<Layout::ColumnMajor> Convolution::backward(Matrix<Layout::ColumnMajor>& dL, const ActivationFunction& activation_function, const Matrix<Layout::ColumnMajor>& a_prev, float learning_rate);

void Convolution::backward(float learning_rate, Matrix<Layout::ColumnMajor>& dL, const ActivationFunction& activation_function, const Matrix<Layout::ColumnMajor>& a_prev);
