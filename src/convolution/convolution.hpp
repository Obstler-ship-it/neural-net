// Faltung mit einem quadratischen Kernel

#pragma once

#include <csignal>
#include <vector>
#include "../matrix/matrix.hpp"

struct Kernels{
    Kernels(size_t kernel_size, size_t number_kernels)
        :KERNEL_SIZE_SQUARED(kernel_size * kernel_size){
            weights.reserve(KERNEL_SIZE_SQUARED * number_kernels);
        }

    std::vector<float> weights;

    const size_t KERNEL_SIZE_SQUARED;

    float operator() (size_t kernel, size_t weight){
         return weights[kernel * KERNEL_SIZE_SQUARED + weight];
    }
};

class Convolution{
    public:
        Convolution(size_t kernel_size, size_t number_kernels)
            :KERNEL_SIZE(kernel_size),
             NUMBER_KERNELS(number_kernels),
             kernels(kernel_size, number_kernels) {}

        const size_t KERNEL_SIZE;
        const size_t NUMBER_KERNELS;

        Kernels kernels;

        Matrix<Layout::ColumnMajor> forward(const std::vector<float>& data, std::span<const size_t> batch_indizes);



};
