#pragma once

#include "activation.hpp"
#include <vector>

class ReLu: public ActivationFunction {
    public:
        std::vector<float> activate(const std::vector<float>& vector) const override{
            std::vector<float> result;
            result.reserve(vector.size());

            for (float number: vector) {
                result.push_back(std::max(0.0f, number));
            }

            return result;
        }

        std::vector<float> derivative(const std::vector<float>& vector) const override{
            std::vector<float> result;
            result.reserve(vector.size());

            for (float number: vector) {
                if (number > 0)
                    result.push_back(1);
                else
                    result.push_back(0);
            }

            return result;
        }
};
