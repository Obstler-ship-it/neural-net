#include "convolution.hpp"
#include "matrix/matrix.hpp"
#include "mec/mec.hpp"
#include <cstddef>
#include <stdexcept>
#include <vector>

Matrix<Layout::ColumnMajor> Convolution::forward(const Images& input, std::span<const size_t> batch_indizes, const ActivationFunction& activation_function){
    if (KERNEL_SIZE == 0 || NUMBER_KERNELS == 0 ||
        input.rows_per_image < KERNEL_SIZE || input.columns_per_image < KERNEL_SIZE ||
        input.number_images == 0 || batch_indizes.empty())
        throw std::invalid_argument("Ungültige Bild-, Kernel- oder Batchgröße");

    const size_t mec_image_size = (input.rows_per_image - KERNEL_SIZE + 1)
                               * input.columns_per_image * KERNEL_SIZE;
    if (input.data.size() != mec_image_size * input.number_images)
        throw std::invalid_argument("MEC-Daten passen nicht zu den Bilddimensionen");
    for (size_t image_index : batch_indizes)
        if (image_index >= input.number_images)
            throw std::out_of_range("Batch-Index außerhalb der Bilddaten");

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
    activation_ = result;
    input_rows_ = input.rows_per_image;
    input_columns_ = input.columns_per_image;
    return result;
}

Matrix<Layout::ColumnMajor> Convolution::backward(
    Matrix<Layout::ColumnMajor>& dL, const ActivationFunction& activation_function,
    const Matrix<Layout::ColumnMajor>& a_prev, float learning_rate) {
    Matrix<Layout::ColumnMajor> result(0, 0, Initialization::Zero);
    backward_impl(dL, activation_function, a_prev, learning_rate, &result);
    return result;
}

void Convolution::backward(
    float learning_rate, Matrix<Layout::ColumnMajor>& dL,
    const ActivationFunction& activation_function,
    const Matrix<Layout::ColumnMajor>& a_prev) {
    backward_impl(dL, activation_function, a_prev, learning_rate, nullptr);
}

void Convolution::backward_impl(
    Matrix<Layout::ColumnMajor>& dL, const ActivationFunction& activation_function,
    const Matrix<Layout::ColumnMajor>& a_prev, float learning_rate,
    Matrix<Layout::ColumnMajor>* input_gradient) {
    if (activation_.columns == 0)
        throw std::logic_error("backward benötigt einen vorherigen forward-Aufruf");
    if (dL.rows != activation_.rows || dL.columns != activation_.columns ||
        dL.data.size() != activation_.data.size() ||
        a_prev.rows != input_rows_ * input_columns_ ||
        a_prev.columns != activation_.columns ||
        a_prev.data.size() != a_prev.rows * a_prev.columns)
        throw std::invalid_argument("Gradient oder Eingabe passen nicht zum letzten forward-Batch");

    Matrix<Layout::RowMajor> kernel_gradient(KERNEL_SIZE_SQUARED, NUMBER_KERNELS, Initialization::Zero);
    std::vector<float> bias_gradient(NUMBER_KERNELS, 0.0f);
    if (input_gradient)
        input_gradient->resize(a_prev.rows, a_prev.columns);

    activation_function.backward(dL, activation_);

    const size_t output_rows = input_rows_ - KERNEL_SIZE + 1;
    const size_t output_columns = input_columns_ - KERNEL_SIZE + 1;
    for (size_t batch = 0; batch < dL.columns; ++batch) {
        for (size_t y = 0; y < output_rows; ++y) {
            for (size_t x = 0; x < output_columns; ++x) {
                const size_t output_offset = batch * dL.rows
                    + (y * output_columns + x) * NUMBER_KERNELS;
                for (size_t kernel = 0; kernel < NUMBER_KERNELS; ++kernel)
                    bias_gradient[kernel] += dL.data[output_offset + kernel];

                for (size_t idx = 0; idx < KERNEL_SIZE_SQUARED; ++idx) {
                    // MEC speichert innerhalb des Kernels zuerst die vertikale Richtung.
                    const size_t input_offset = batch * a_prev.rows
                        + (y + idx % KERNEL_SIZE) * input_columns_ + x + idx / KERNEL_SIZE;
                    const float pixel = a_prev.data[input_offset];
                    float pixel_gradient = 0.0f;
                    for (size_t kernel = 0; kernel < NUMBER_KERNELS; ++kernel) {
                        const float gradient = dL.data[output_offset + kernel];
                        kernel_gradient(idx, kernel) += pixel * gradient;
                        if (input_gradient)
                            pixel_gradient += Kernels(idx, kernel) * gradient;
                    }
                    // Beiträge überlappender Ausschnitte sammeln, bevor Gewichte geändert werden.
                    if (input_gradient)
                        input_gradient->data[input_offset] += pixel_gradient;
                }
            }
        }
    }

    const float scale = learning_rate / static_cast<float>(dL.columns);
    for (size_t i = 0; i < Kernels.data.size(); ++i)
        Kernels.data[i] -= scale * kernel_gradient.data[i];
    for (size_t kernel = 0; kernel < NUMBER_KERNELS; ++kernel)
        Bias.data[kernel] -= scale * bias_gradient[kernel];
}
