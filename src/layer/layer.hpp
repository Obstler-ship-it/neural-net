// Layer mit Neuronen und Gewichten
#pragma once

#include "../matrix/matrix.hpp"
#include <cstddef>
#include <sys/types.h>

class Layer{
    public:
        Layer(size_t input_size, size_t output_size, uint batch_size)
        : input_size(input_size),
          output_size(output_size),
          W(output_size, input_size, Initialization::Random),
          a(input_size, batch_size, Initialization::Uninitialized),
          b(output_size, 0, Initialization::Random),
          dW(output_size, input_size, Initialization::Zero) {}

        size_t input_size;
        size_t output_size;
        uint batch_size;

        Matrix<Layout::RowMajor> W;
        Matrix<Layout::ColumnMajor> a;
        Matrix<Layout::ColumnMajor> b;

        Matrix<Layout::ColumnMajor> dW;

        void forward(const Matrix<Layout::ColumnMajor>& activation);
        void backward(const Matrix<Layout::ColumnMajor>& dL);
};
