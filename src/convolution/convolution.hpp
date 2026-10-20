// Faltung mit einem quadratischen Kernel

#pragma once

#include <span>
#include "../matrix/matrix.hpp"
#include "../activation/activation.hpp"
#include "../mec/mec.hpp"

class Convolution{
    public:
        Convolution(size_t kernel_size, size_t number_kernels)
            :KERNEL_SIZE(kernel_size),
             KERNEL_SIZE_SQUARED(kernel_size * kernel_size),
             NUMBER_KERNELS(number_kernels),
             Bias(1, number_kernels, Initialization::Random),
             Kernels(KERNEL_SIZE_SQUARED, number_kernels, Initialization::Random){}

        const size_t KERNEL_SIZE;
        const size_t KERNEL_SIZE_SQUARED;
        const size_t NUMBER_KERNELS;

        Matrix<Layout::RowMajor> Bias;
        Matrix<Layout::RowMajor> Kernels;

        // Forward für die Faltung eines mec std::vectors
        Matrix<Layout::ColumnMajor> forward(const Images& input, std::span<const size_t> batch_indizes, const ActivationFunction& activation_function);

        // a_prev: ursprüngliche Bilder, zeilenweise abgeflacht, eine Spalte je
        // Batch-Eintrag in derselben Reihenfolge wie batch_indizes in forward.
        // Benötigt einen vorherigen forward-Aufruf. dL wird zu dL/dz geändert.
        // Parametergradienten werden über den Batch gemittelt, Eingabegradienten nicht.
        //backward mit Ausgabe eines Gradienten
        Matrix<Layout::ColumnMajor> backward(Matrix<Layout::ColumnMajor>& dL, const ActivationFunction& activation_function, const Matrix<Layout::ColumnMajor>& a_prev, float learning_rate);

        //backward ohne Ausgabe für die letzte Layer
        void backward(float learning_rate, Matrix<Layout::ColumnMajor>& dL, const ActivationFunction& activation_function, const Matrix<Layout::ColumnMajor>& a_prev);
    private:
        Matrix<Layout::ColumnMajor> activation_{0, 0, Initialization::Zero};
        size_t input_rows_ = 0;
        size_t input_columns_ = 0;

        void backward_impl(Matrix<Layout::ColumnMajor>& dL,
                           const ActivationFunction& activation_function,
                           const Matrix<Layout::ColumnMajor>& a_prev,
                           float learning_rate,
                           Matrix<Layout::ColumnMajor>* input_gradient);
};
