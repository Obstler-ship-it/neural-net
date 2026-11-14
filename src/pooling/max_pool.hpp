#pragma once

#include "../matrix/matrix.hpp"
#include <stdexcept>
#include <vector>

// 2x2-Max-Pooling, Schrittweite 2, ohne Padding.
// Layout wie Convolution: (y * Breite + x) * Kanäle + Kanal; ein Bild je Spalte.
// Bei ungeraden Dimensionen wird die letzte Zeile/Spalte ausgelassen.
class MaxPool2D {
public:
    static size_t output_size(size_t rows, size_t columns, size_t channels) {
        if (rows < 2 || columns < 2 || channels == 0)
            throw std::invalid_argument("Feature-Maps müssen mindestens 2x2 groß sein");
        return (rows / 2) * (columns / 2) * channels;
    }

    Matrix<Layout::ColumnMajor> forward(const Matrix<Layout::ColumnMajor>& input,
                                       size_t rows, size_t columns, size_t channels) {
        input_size_ = 0;
        const size_t output_rows = output_size(rows, columns, channels);
        if (input.rows != rows * columns * channels || input.columns == 0 ||
            input.data.size() != input.rows * input.columns)
            throw std::invalid_argument("Pooling-Eingabe passt nicht zu den Feature-Maps");
        Matrix<Layout::ColumnMajor> result(output_rows, input.columns, Initialization::Zero);
        maxima_.resize(output_rows * input.columns);
        size_t destination = 0;
        for (size_t batch = 0; batch < input.columns; ++batch) {
            for (size_t y = 0; y < rows / 2; ++y) {
                for (size_t x = 0; x < columns / 2; ++x) {
                    for (size_t channel = 0; channel < channels; ++channel) {
                        const size_t top_left = batch * input.rows + (2 * y * columns + 2 * x) * channels + channel;
                        size_t best = top_left;
                        float best_value = input.data[top_left];
                        for (size_t offset : {channels, columns * channels, (columns + 1) * channels}) {
                            const float candidate = input.data[top_left + offset];
                            const bool larger = candidate > best_value;
                            best = larger ? top_left + offset : best;
                            best_value = larger ? candidate : best_value;
                        }
                        result.data[destination] = best_value;
                        maxima_[destination++] = best;
                    }
                }
            }
        }
        input_size_ = input.rows;
        batch_size_ = input.columns;
        output_size_ = output_rows;
        return result;
    }

    Matrix<Layout::ColumnMajor> backward(const Matrix<Layout::ColumnMajor>& gradient) const {
        if (input_size_ == 0)
            throw std::logic_error("Pooling-backward benötigt vorher forward");
        if (gradient.rows != output_size_ || gradient.columns != batch_size_ ||
            gradient.data.size() != maxima_.size())
            throw std::invalid_argument("Pooling-Gradient hat falsche Dimensionen");
        Matrix<Layout::ColumnMajor> result(input_size_, batch_size_, Initialization::Zero);
        for (size_t i = 0; i < maxima_.size(); ++i)
            result.data[maxima_[i]] = gradient.data[i];
        return result;
    }

private:
    std::vector<size_t> maxima_;
    size_t input_size_ = 0;
    size_t output_size_ = 0;
    size_t batch_size_ = 0;
};
