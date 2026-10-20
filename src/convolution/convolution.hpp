// Faltung mit einem quadratischen Kernel

#pragma once

//#include <vector>
#include "../matrix/matrix.hpp"
#include "../activation/activation.hpp"
#include "../mec/mec.hpp"

class Convolution{
    public:
        Convolution(size_t kernel_size, size_t number_kernels)
            :KERNEL_SIZE(kernel_size),
             KERNEL_SIZE_SQUARED(kernel_size * kernel_size),
             NUMBER_KERNELS(number_kernels),
             Kernels(KERNEL_SIZE_SQUARED, number_kernels, Initialization::Random),
             Bias(1, number_kernels, Initialization::Random){}

        const size_t KERNEL_SIZE;
        const size_t KERNEL_SIZE_SQUARED;
        const size_t NUMBER_KERNELS;

        Matrix<Layout::RowMajor> Bias;
        Matrix<Layout::RowMajor> Kernels;

        // Forward für die Faltung eines mec std::vectors
        Matrix<Layout::ColumnMajor> forward(const Images& input, std::span<const size_t> batch_indizes, const ActivationFunction& activation_function);

        //backward mit Ausgabe eines Gradienten
        Matrix<Layout::ColumnMajor> backward(Matrix<Layout::ColumnMajor>& dL, const ActivationFunction& activation_function, const Matrix<Layout::ColumnMajor>& a_prev, float learning_rate);

        //backward ohne Ausgabe für die letzte Layer
        void backward(float learning_rate, Matrix<Layout::ColumnMajor>& dL, const ActivationFunction& activation_function, const Matrix<Layout::ColumnMajor>& a_prev);
};
