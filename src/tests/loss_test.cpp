#include "../loss/mse.hpp"
#include "../loss/cross_entropy.hpp"
#include "../softmax/softmax.hpp"
#include <cassert>
#include <cmath>
#include <iostream>

int main() {
    MeanSquareError mse;
    Matrix<Layout::ColumnMajor> p(3, 2, Initialization::Zero);
    p.data = {0.2f, 0.3f, 0.5f, 0.6f, 0.1f, 0.3f};
    const std::vector<int> labels{2, 0};
    // Includes the errors of the non-target classes; summed over both examples.
    assert(std::abs(mse.forward(p, labels) - 0.64f / 3.0f) < 1e-6f);

    Matrix<Layout::ColumnMajor> logits(3, 2, Initialization::Zero);
    logits.data = {-0.7f, 0.2f, 1.1f, 0.8f, -0.3f, 0.4f};
    auto probabilities = logits;
    softmax::forward(probabilities);
    const auto gradient = softmax::backward(mse.backward(probabilities, labels), probabilities);
    CrossEntropy ce;
    auto ce_gradient = probabilities;
    for (size_t col = 0; col < labels.size(); ++col)
        ce_gradient(labels[col], col) -= 1.0f;

    constexpr float h = 1e-3f;
    for (size_t i = 0; i < logits.data.size(); ++i) {
        auto plus = logits;
        auto minus = logits;
        plus.data[i] += h;
        minus.data[i] -= h;
        softmax::forward(plus);
        softmax::forward(minus);
        const float numerical = (mse.forward(plus, labels) - mse.forward(minus, labels)) / (2 * h);
        assert(std::abs(gradient.data[i] - numerical) < 1e-4f);
        const float numerical_ce = (ce.forward(plus, labels) - ce.forward(minus, labels)) / (2 * h);
        assert(std::abs(ce_gradient.data[i] - numerical_ce) < 2e-4f);
    }
    Matrix<Layout::ColumnMajor> wrong_shape(1, 2, Initialization::Zero);
    bool rejected = false;
    try { softmax::backward(wrong_shape, probabilities); }
    catch (const std::invalid_argument&) { rejected = true; }
    assert(rejected);
    std::cout << "MSE and cross-entropy gradient checks passed.\n";
}
