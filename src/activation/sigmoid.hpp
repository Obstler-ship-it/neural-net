#pragma once

#include "activation.hpp"
#include <cmath>
#include <vector>
class Sigmoid: public ActivationFunction {
    public:
        std::vector<float> activate(const std::vector<float>& vector) const override {
            std::vector<float> result;
            result.reserve(vector.size());

            for (float number: vector) {
                result.push_back(1.0f/(1.0f + std::exp(-number)));
            }

            return result;
        }

        std::vector<float> derivative(const std::vector<float>& vector) const override{
            std::vector<float> result;
            result.reserve(vector.size());

            for (float number: vector) {
                result.push_back(number * (1 - number));
            }

            return result;
        }
};
