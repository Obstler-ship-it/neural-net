#pragma once

#include <vector>

class ActivationFunction {
public:
    virtual std::vector<float> activate(const std::vector<float>& vector) const = 0;
    virtual std::vector<float> derivative(const std::vector<float>& vector) const = 0;
    virtual ~ActivationFunction() = default;
};
