#pragma once

#include "../matrix/matrix.hpp"
#include <algorithm>
#include <vector>

namespace ARG_MAX{

    inline float count(const Matrix<Layout::ColumnMajor>& y_hat, const std::vector<int>& y, size_t labels) {

        if (y_hat.columns != y.size())
            throw std::invalid_argument("Batch und Anzahl Labels haben unterschiedliche Größe!");

        float count = 0;

        for (size_t i=0; i< y_hat.columns; i++){
            auto first = y_hat.data.begin() + i * labels;
            auto max = std::max_element(first, first + labels);
            [[likely]] //hoffentlich likely
            if (max - first == y[i])
                ++count;
        }

    return count / y_hat.columns;
    }
};
