#pragma once

#include "../matrix/matrix.hpp"
#include <cmath>

//TODO

inline void softmax(Matrix<Layout::ColumnMajor> z){
    float sum = 0.0f;

    for(float number: z.data){
        sum += std::exp(number);
    }

    for(float& number: z.data){
        number = std::exp(number) / sum;
    }
}

inline
