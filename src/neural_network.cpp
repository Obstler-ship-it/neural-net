//
#include "neural_network.hpp"
#include "layer/layer.hpp"
#include "matrix/matrix.hpp"
#include "softmax/softmax.hpp"
#include <iostream>
#include <vector>

float NeuralNetwork::forward_train(const Matrix<Layout::ColumnMajor>& input, const std::vector<int>& labels){
    Layers[0].forward(input, *activation_);
    for (size_t i=1; i < Layers.size() - 1; i++){
        Layers[i].forward(Layers[i-1].a, *activation_);
    }
    Layers[Layers.size() - 1].forward(Layers[Layers.size() - 2].a);
    softmax::forward(Layers.back().a);
    return loss_->forward(Layers.back().a, labels);
}

void NeuralNetwork::backward(const Matrix<Layout::ColumnMajor>& input, const std::vector<int>& y){
    auto& y_hat = Layers.back().a;
    auto gradient = loss_->backward(y_hat, y);
    gradient = softmax::backward(gradient , y, y_hat);
    Layers[Layers.size() - 1].backward(gradient, Layers[Layers.size() - 2].a, learning_rate);
    for (size_t i=Layers.size() - 2; i > 0; i--){
        gradient = Layers[i].backward(gradient, *activation_, Layers[i-1].a, learning_rate);
    }
    Layers[0].backward(gradient, *activation_, input, learning_rate);
}

void NeuralNetwork::train(std::vector<std::vector<float>>, std::vector<int>){

}
