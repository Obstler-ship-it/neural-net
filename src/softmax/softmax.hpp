#pragma once

#include <vector>
#include <cmath>

inline void softmax(std::vector<float>& vector){
    float sum = 0.0f;

    for(float number: vector){
        sum += std::exp(number);
    }

    for(float& number: vector){
        number = std::exp(number) / sum;
    }
}