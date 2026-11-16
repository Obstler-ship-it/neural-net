// Layer mit Neuronen und Gewichten
#pragma once

#include "../matrix/matrix.hpp"
#include "../activation/activation.hpp"
#include <cstddef>
#include <cmath>

template<class OptimizerType>
class Layer{
    public:
        Layer(size_t input_size, size_t output_size, uint batch_size, const ActivationFunction& activation_function)
        : input_size(input_size),
          output_size(output_size),
          batch_size(batch_size),
          W(output_size, input_size, Initialization::Random),
          a(input_size, batch_size, Initialization::Uninitialized),
          b(output_size, 1, Initialization::Zero),
          db(output_size, 1, Initialization::Uninitialized),
          dW(output_size, input_size, Initialization::Uninitialized) {
            //initialisierung abhängig von der Aktivierungsfunktion
            W *= activation_function.scale(input_size, output_size);
        }

        //Enthält die Ausgabe nach der Layer und Aktivierungsfunktion
        Matrix<Layout::ColumnMajor> a; //public

        void rescale_for(const ActivationFunction& activation_function) {
            W *= activation_function.scale(input_size, output_size)
                / std::sqrt(6.0f / static_cast<float>(input_size));
        }

    private:
        size_t input_size;
        size_t output_size;
        size_t batch_size;

        Matrix<Layout::RowMajor> W;
        Matrix<Layout::ColumnMajor> b;

        Matrix<Layout::ColumnMajor> db;
        Matrix<Layout::ColumnMajor> dW;

        OptimizerType optimizer; //Enhält je nach Verfahren

    public:
        // Berechnet die Layer weiter mit Aktivierungsfunktion
        void forward(const Matrix<Layout::ColumnMajor>& activation, const ActivationFunction& activation_function);

        // Berechnet die Layer ohne Aktivierungsfunktion
        void forward(const Matrix<Layout::ColumnMajor>& activation);

        // Berechnet die nächste Ableitung und aktualisiert die Gewichte mit Aktivierungsfunktion
        Matrix<Layout::ColumnMajor> backward(Matrix<Layout::ColumnMajor>& dL, const ActivationFunction& activation_function, const Matrix<Layout::ColumnMajor>& a_prev);

        // Berechnet die nächste Ableitung und aktualisiert die Gewichte ohne Aktivierungsfunktion
        Matrix<Layout::ColumnMajor> backward(Matrix<Layout::ColumnMajor>& dL, const Matrix<Layout::ColumnMajor>& a_prev);

        // Aktualisiert die Gewichte ohne die nächste Ableitung zu berechnen
        void backward(Matrix<Layout::ColumnMajor>& dL, const Matrix<Layout::ColumnMajor>& a_prev, const ActivationFunction& activation_function);
};

#include "layer.cpp"
