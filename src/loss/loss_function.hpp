#pragma once

#include <vector>

class LossFunction{
    public:
        virtual float loss(const std::vector<float>& vector) const = 0;
        virtual std::vector<float> derivative(const std::vector<float>& vector) const = 0;
        virtual ~LossFunction() = default;
}
