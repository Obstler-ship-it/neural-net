#pragma once

#include <cmath>
#include <vector>

inline std::vector<float> sigmoid(const std::vector<float> vector){
    std::vector<float> result;
    result.reserve(vector.size());
    for(float number: vector){
        result.push_back(1.0f/(1.0f + std::exp(-number)));
    }
    return result;
}