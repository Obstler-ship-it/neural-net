#pragma once

#include "loss_function.hpp"
#include <vector>

class CrossEntropy: public LossFunction{
    public:
        std::vector<float> loss(const std::vector<float>& vector) const override{
            std::vector<float> result;
            result.reserve(vector.size());

        

            return result;
        }
        }
}