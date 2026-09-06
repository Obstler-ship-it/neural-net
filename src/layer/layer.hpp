#pragma once

#include <cstddef>
#include <vector>


class Layer{
    public:
    Layer(std::size_t rows, std::size_t columns);
    float& operator() (int row, int column);
    std::vector<float> dot(const std::vector<float>);

    private:
    std::vector<float> layer;
    std::size_t rows = 0;
    std::size_t columns = 0;
};