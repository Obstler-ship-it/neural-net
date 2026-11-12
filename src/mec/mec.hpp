//  Memory-efficient Convolution for Deep Neural Network

#pragma once

#include <cstddef>
#include <stdexcept>
#include <vector>

struct Dataset{
    std::vector<float> data;
    size_t number_images;
    size_t columns_per_image;
    size_t rows_per_image;
};

/*
 * @brief Speichert ein Image linear in einen Vector um möglichst schnell die Faltung mit Kernels zu berechnen
 * ähnlich zu im2col nur mit weniger Speicherverbrauch
 * @param Images, rows, columns, kernel_size k x k
 * @return std::vector<float> data, size_t number_images, size_t rows_per_image
 */
Dataset mec(const std::vector<std::vector<float>>& images, size_t rows, size_t columns, size_t kernel_size);

inline Dataset mec(const std::vector<std::vector<float>>& images, size_t rows, size_t columns, size_t kernel_size){

    if (kernel_size == 0 || kernel_size > rows || kernel_size > columns)
        throw std::invalid_argument("Ungültige Kernelgröße");

    Dataset result;
    result.number_images = images.size();
    result.columns_per_image = columns;
    result.rows_per_image = rows;
    result.data.reserve(result.number_images * rows * columns * kernel_size);
    size_t index;

    for (const auto& image:images){
        if (image.size() != rows * columns)
            throw std::invalid_argument("Falsche Bildgröße");
        for (size_t row=0; row < rows - kernel_size + 1; ++row){
            for(size_t column=0; column < columns; ++column){
                for(size_t i=0; i < kernel_size; ++i){
                    index = (row + i) * columns + column;
                    result.data.push_back(image[index]);
                }
            }
        }
    }

    return result;
}
