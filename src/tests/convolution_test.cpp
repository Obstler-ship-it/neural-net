#include "../convolution/convolution.hpp"
#include "../activation/sigmoid.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
void close(float actual, float expected, const char* label) {
    if (!std::isfinite(actual) || std::abs(actual - expected) > 0.003f * (1.0f + std::abs(expected)))
        throw std::runtime_error(std::string(label) + ": " + std::to_string(actual)
                                 + " != " + std::to_string(expected));
}

class Identity : public ActivationFunction {
public:
    void forward(Matrix<Layout::ColumnMajor>&) const override {}
    void backward(Matrix<Layout::ColumnMajor>&, const Matrix<Layout::ColumnMajor>&) const override {}
};

template<class Exception, class Function>
void expect_throw(Function function) {
    try { function(); }
    catch (const Exception&) { return; }
    throw std::runtime_error("Erwartete Ausnahme fehlt");
}

void check_gradients(size_t kernel_size, const ActivationFunction& activation) {
    constexpr size_t rows = 3, columns = 4, kernels = 3;
    const std::vector<size_t> indices{2, 0, 2}; // Reihenfolge und wiederholte Bilder
    std::vector<std::vector<float>> images(3, std::vector<float>(rows * columns));
    for (size_t image = 0; image < images.size(); ++image)
        for (size_t pixel = 0; pixel < images[image].size(); ++pixel)
            images[image][pixel] = (static_cast<int>((pixel * 7 + image * 3) % 17) - 8) * 0.07f;

    Convolution original(kernel_size, kernels);
    for (size_t i = 0; i < original.Kernels.data.size(); ++i)
        original.Kernels.data[i] = (static_cast<int>(i % 7) - 3) * 0.09f;
    original.Bias.data = {-0.1f, 0.05f, 0.2f};
    auto output = original.forward(mec(images, rows, columns, kernel_size), indices, activation);

    // Unabhängige direkte Faltung prüft das MEC- und Ausgabelayout.
    const size_t out_rows = rows - kernel_size + 1, out_columns = columns - kernel_size + 1;
    Matrix<Layout::ColumnMajor> expected(output.rows, output.columns, Initialization::Zero);
    for (size_t b = 0; b < indices.size(); ++b)
        for (size_t y = 0; y < out_rows; ++y)
            for (size_t x = 0; x < out_columns; ++x)
                for (size_t k = 0; k < kernels; ++k) {
                    float value = original.Bias.data[k];
                    for (size_t dx = 0; dx < kernel_size; ++dx)
                        for (size_t dy = 0; dy < kernel_size; ++dy)
                            value += images[indices[b]][(y + dy) * columns + x + dx]
                                   * original.Kernels(dx * kernel_size + dy, k);
                    expected((y * out_columns + x) * kernels + k, b) = value;
                }
    activation.forward(expected);
    for (size_t i = 0; i < output.data.size(); ++i)
        close(output.data[i], expected.data[i], "forward");

    Matrix<Layout::ColumnMajor> previous(rows * columns, indices.size(), Initialization::Uninitialized);
    std::vector<std::vector<float>> selected;
    for (size_t index : indices) {
        previous.data.insert(previous.data.end(), images[index].begin(), images[index].end());
        selected.push_back(images[index]);
    }
    Matrix<Layout::ColumnMajor> upstream(output.rows, output.columns, Initialization::Zero);
    for (size_t i = 0; i < upstream.data.size(); ++i)
        upstream.data[i] = (static_cast<int>(i % 11) - 5) * 0.11f;

    auto with_gradient = original;
    auto without_gradient = original;
    auto gradient = upstream;
    auto gradient_void = upstream;
    constexpr float rate = 0.25f;
    const auto input_gradient = with_gradient.backward(gradient, activation, previous, rate);
    without_gradient.backward(rate, gradient_void, activation, previous);
    if (input_gradient.rows != previous.rows || input_gradient.columns != previous.columns)
        throw std::runtime_error("Falsche Eingabegradient-Dimensionen");
    for (size_t i = 0; i < gradient.data.size(); ++i)
        close(gradient.data[i], gradient_void.data[i], "dL-Overloads");
    for (size_t i = 0; i < original.Kernels.data.size(); ++i)
        close(with_gradient.Kernels.data[i], without_gradient.Kernels.data[i], "Kernel-Overloads");
    for (size_t i = 0; i < kernels; ++i)
        close(with_gradient.Bias.data[i], without_gradient.Bias.data[i], "Bias-Overloads");

    const std::vector<size_t> selected_indices{0, 1, 2};
    auto objective = [&]() {
        auto prediction = original.forward(mec(selected, rows, columns, kernel_size), selected_indices, activation);
        double value = 0.0;
        for (size_t i = 0; i < prediction.data.size(); ++i)
            value += static_cast<double>(prediction.data[i]) * upstream.data[i];
        return value;
    };
    auto derivative = [&](float& parameter) {
        const float saved = parameter;
        constexpr float epsilon = 0.001f;
        parameter = saved + epsilon;
        const double plus = objective();
        parameter = saved - epsilon;
        const double minus = objective();
        parameter = saved;
        return static_cast<float>((plus - minus) / (2.0 * epsilon));
    };
    for (size_t i = 0; i < original.Kernels.data.size(); ++i) {
        const float numerical = derivative(original.Kernels.data[i]) / indices.size();
        close((original.Kernels.data[i] - with_gradient.Kernels.data[i]) / rate, numerical, "dKernel");
    }
    for (size_t i = 0; i < kernels; ++i) {
        const float numerical = derivative(original.Bias.data[i]) / indices.size();
        close((original.Bias.data[i] - with_gradient.Bias.data[i]) / rate, numerical, "dBias");
    }
    for (size_t b = 0; b < selected.size(); ++b)
        for (size_t pixel = 0; pixel < rows * columns; ++pixel)
            close(input_gradient(pixel, b), derivative(selected[b][pixel]), "dInput");
}

void check_errors() {
    Identity activation;
    Convolution convolution(2, 2);
    Matrix<Layout::ColumnMajor> previous(9, 1, Initialization::Zero);
    Matrix<Layout::ColumnMajor> gradient(8, 1, Initialization::Zero);
    expect_throw<std::logic_error>([&] { convolution.backward(gradient, activation, previous, 0.1f); });
    const auto input = mec({std::vector<float>(9, 1.0f)}, 3, 3, 2);
    const std::vector<size_t> valid{0}, invalid{1};
    expect_throw<std::invalid_argument>([&] { convolution.forward(input, {}, activation); });
    expect_throw<std::out_of_range>([&] { convolution.forward(input, invalid, activation); });
    convolution.forward(input, valid, activation);
    previous.rows = 8;
    expect_throw<std::invalid_argument>([&] { convolution.backward(gradient, activation, previous, 0.1f); });
    previous.rows = 9;
    gradient.data.pop_back();
    expect_throw<std::invalid_argument>([&] { convolution.backward(0.1f, gradient, activation, previous); });
}
}

int main() {
    Identity identity;
    Sigmoid sigmoid;
    for (size_t kernel_size : {1, 2, 3}) {
        check_gradients(kernel_size, identity);
        check_gradients(kernel_size, sigmoid);
    }
    check_errors();
    std::cout << "Alle Convolution-Gradiententests erfolgreich.\n";
}
