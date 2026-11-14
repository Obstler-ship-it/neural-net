#include "convolution.hpp"
#include "../matrix/matrix.hpp"
#include "../mec/mec.hpp"
#include "../matrix/gemm.hpp"
#include "../parallel.hpp"
#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <vector>

const Matrix<Layout::ColumnMajor>& Convolution::forward(const Images& input, std::span<const size_t> batch_indizes, const ActivationFunction& activation_function){
    if (INPUT_CHANNELS != 1 ||
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

    activation_.resize(output_size, batch_indizes.size());
    auto& result = activation_;

    // Anzahl gespeicherter MEC-Werte pro Bild
    const size_t image_size = input.data.size() / input.number_images;
    const size_t strip_size = input.columns_per_image * KERNEL_SIZE;

    const size_t workers = parallel::worker_count(batch_indizes.size(), output_size * KERNEL_SIZE_SQUARED);
    parallel::for_ranges(batch_indizes.size(), workers, [&](size_t, size_t begin, size_t end) {
        for (size_t batch = begin; batch < end; ++batch) {
            float* output = result.data.data() + batch * output_size;
            for (size_t position = 0; position < output_rows * output_columns; ++position)
                std::copy(Bias.data.begin(), Bias.data.end(), output + position * NUMBER_KERNELS);
            for (size_t output_row = 0; output_row < output_rows; ++output_row) {
                const float* strip = input.data.data() + batch_indizes[batch] * image_size + output_row * strip_size;
                matrix_detail::multiply_add(strip, Kernels.data.data(), output + output_row * output_columns * NUMBER_KERNELS,
                                            output_columns, KERNEL_SIZE_SQUARED, NUMBER_KERNELS, KERNEL_SIZE);
            }
        }
    });

    activation_function.forward(result);
    input_rows_ = input.rows_per_image;
    input_columns_ = input.columns_per_image;
    return result;
}

const Matrix<Layout::ColumnMajor>& Convolution::forward(
    const Matrix<Layout::ColumnMajor>& input, size_t rows, size_t columns,
    const ActivationFunction& activation_function) {
    if (rows < KERNEL_SIZE || columns < KERNEL_SIZE || input.columns == 0 ||
        input.rows != rows * columns * INPUT_CHANNELS ||
        input.data.size() != input.rows * input.columns)
        throw std::invalid_argument("Convolution-Eingabe passt nicht zu den Feature-Maps");

    const size_t output_rows = rows - KERNEL_SIZE + 1;
    const size_t output_columns = columns - KERNEL_SIZE + 1;
    const size_t positions = output_rows * output_columns;
    activation_.resize(positions * NUMBER_KERNELS, input.columns);
    auto& result = activation_;
    const size_t workers = parallel::worker_count(input.columns, positions * Kernels.rows * NUMBER_KERNELS);
    parallel::for_ranges(input.columns, workers, [&](size_t, size_t begin, size_t end) {
        for (size_t batch = begin; batch < end; ++batch) {
            float* output = result.data.data() + batch * result.rows;
            for (size_t position = 0; position < positions; ++position)
                std::copy(Bias.data.begin(), Bias.data.end(), output + position * NUMBER_KERNELS);
            for (size_t output_row = 0; output_row < output_rows; ++output_row)
                for (size_t position = 0; position < KERNEL_SIZE_SQUARED; ++position) {
                    const size_t input_offset = batch * input.rows
                        + ((output_row + position % KERNEL_SIZE) * columns + position / KERNEL_SIZE) * INPUT_CHANNELS;
                    matrix_detail::multiply_add(input.data.data() + input_offset,
                                                Kernels.data.data() + position * INPUT_CHANNELS * NUMBER_KERNELS,
                                                output + output_row * output_columns * NUMBER_KERNELS,
                                                output_columns, INPUT_CHANNELS, NUMBER_KERNELS, INPUT_CHANNELS);
                }
        }
    });

    activation_function.forward(result);
    input_rows_ = rows;
    input_columns_ = columns;
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
        a_prev.rows != input_rows_ * input_columns_ * INPUT_CHANNELS ||
        a_prev.columns != activation_.columns ||
        a_prev.data.size() != a_prev.rows * a_prev.columns)
        throw std::invalid_argument("Gradient oder Eingabe passen nicht zum letzten forward-Batch");

    if (input_gradient)
        input_gradient->resize(a_prev.rows, a_prev.columns);

    activation_function.backward(dL, activation_);

    const size_t output_rows = input_rows_ - KERNEL_SIZE + 1;
    const size_t output_columns = input_columns_ - KERNEL_SIZE + 1;
    const size_t positions = output_rows * output_columns;
    const size_t workers = parallel::worker_count(dL.columns, positions * Kernels.rows * NUMBER_KERNELS);
    std::vector<std::vector<float>> kernel_gradients(workers, std::vector<float>(Kernels.data.size(), 0.0f));
    std::vector<std::vector<float>> bias_gradients(workers, std::vector<float>(NUMBER_KERNELS, 0.0f));
    std::vector<float> transposed_kernels;
    if (input_gradient) {
        transposed_kernels.resize(Kernels.data.size());
        for (size_t position = 0; position < KERNEL_SIZE_SQUARED; ++position)
            for (size_t kernel = 0; kernel < NUMBER_KERNELS; ++kernel)
                for (size_t channel = 0; channel < INPUT_CHANNELS; ++channel)
                    transposed_kernels[(position * NUMBER_KERNELS + kernel) * INPUT_CHANNELS + channel]
                        = Kernels(position * INPUT_CHANNELS + channel, kernel);
    }
    parallel::for_ranges(dL.columns, workers, [&](size_t worker, size_t begin, size_t end) {
        for (size_t batch = begin; batch < end; ++batch) {
            const float* gradient = dL.data.data() + batch * dL.rows;
            for (size_t position = 0; position < positions; ++position)
                for (size_t kernel = 0; kernel < NUMBER_KERNELS; ++kernel)
                    bias_gradients[worker][kernel] += gradient[position * NUMBER_KERNELS + kernel];
            for (size_t output_row = 0; output_row < output_rows; ++output_row)
                for (size_t position = 0; position < KERNEL_SIZE_SQUARED; ++position) {
                    const size_t input_offset = batch * a_prev.rows
                        + ((output_row + position % KERNEL_SIZE) * input_columns_ + position / KERNEL_SIZE) * INPUT_CHANNELS;
                    const size_t kernel_offset = position * INPUT_CHANNELS * NUMBER_KERNELS;
                    const float* gradient_row = gradient + output_row * output_columns * NUMBER_KERNELS;
                    matrix_detail::multiply_add(a_prev.data.data() + input_offset, gradient_row,
                                                kernel_gradients[worker].data() + kernel_offset,
                                                INPUT_CHANNELS, output_columns, NUMBER_KERNELS, 1, INPUT_CHANNELS);
                    if (input_gradient)
                        matrix_detail::multiply_add(gradient_row, transposed_kernels.data() + kernel_offset,
                                                    input_gradient->data.data() + input_offset,
                                                    output_columns, NUMBER_KERNELS, INPUT_CHANNELS, NUMBER_KERNELS);
                }
        }
    });

    const float scale = learning_rate / static_cast<float>(dL.columns);
    for (size_t weight = 0; weight < Kernels.data.size(); ++weight) {
        float gradient = 0.0f;
        for (size_t worker = 0; worker < workers; ++worker) gradient += kernel_gradients[worker][weight];
        Kernels.data[weight] -= scale * gradient;
    }
    for (size_t kernel = 0; kernel < NUMBER_KERNELS; ++kernel) {
        float gradient = 0.0f;
        for (size_t worker = 0; worker < workers; ++worker) gradient += bias_gradients[worker][kernel];
        Bias.data[kernel] -= scale * gradient;
    }
}
