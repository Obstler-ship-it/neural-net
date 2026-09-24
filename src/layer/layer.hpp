// Layer mit Neuronen und Gewichten
#pragma once

#include "../matrix/matrix.hpp"
#include "../activation/activation.hpp"
#include <cstddef>
#include <cmath>
#include <sys/types.h>

class Layer{
    public:
        Layer(size_t input_size, size_t output_size, uint batch_size)
        : input_size(input_size),
          output_size(output_size),
          batch_size(batch_size),
          W(output_size, input_size, Initialization::Random),
          a(input_size, batch_size, Initialization::Uninitialized),
          b(output_size, 1, Initialization::Zero),
          dW(output_size, input_size, Initialization::Zero) {
            W * std::sqrt(6.0f / static_cast<float>(input_size));
        }

        size_t input_size;
        size_t output_size;
        size_t batch_size;

        Matrix<Layout::RowMajor> W;
        Matrix<Layout::ColumnMajor> a;
        Matrix<Layout::ColumnMajor> b;

        Matrix<Layout::ColumnMajor> dW;

        // Berechnet die Layer weiter mit Aktivierungsfunktion
        void forward(const Matrix<Layout::ColumnMajor>& activation, const ActivationFunction& activation_function);

        // Berechnet die Layer ohne Aktivierungsfunktion
        void forward(const Matrix<Layout::ColumnMajor>& activation);

        // Berechnet die nächste Ableitung und aktualisiert die Gewichte mit Aktivierungsfunktion
        Matrix<Layout::ColumnMajor> backward(Matrix<Layout::ColumnMajor>& dL, const ActivationFunction& activation_function, const Matrix<Layout::ColumnMajor>& a_prev,float learning_rate);

        // Berechnet die nächste Ableitung und aktualisiert die Gewichte ohne Aktivierungsfunktion
        Matrix<Layout::ColumnMajor> backward(Matrix<Layout::ColumnMajor>& dL, const Matrix<Layout::ColumnMajor>& a_prev,float learning_rate);
};
