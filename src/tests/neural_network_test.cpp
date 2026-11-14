#include "../neural_network.hpp"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>

namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
void close(float actual, float expected, const char* message, float tolerance = 0.002f) {
    if (!std::isfinite(actual) || std::abs(actual - expected) > tolerance * (1.0f + std::abs(expected)))
        throw std::runtime_error(std::string(message) + ": " + std::to_string(actual)
                                 + " != " + std::to_string(expected));
}
template<class Exception, class Function>
void expect_throw(Function function) {
    try { function(); }
    catch (const Exception&) { return; }
    throw std::runtime_error("Erwartete Ausnahme fehlt");
}
std::vector<std::vector<float>> images() {
    std::vector<std::vector<float>> result(5, std::vector<float>(56));
    for (size_t i = 0; i < result.size(); ++i)
        for (size_t p = 0; p < result[i].size(); ++p)
            result[i][p] = 0.01f * (1 + (p * p * 7 + p * 3 + i * 11) % 97) + 0.001f * p;
    return result;
}
void initialize(NeuralNetwork& net) {
    for (size_t i = 0; i < net.convolution.Kernels.data.size(); ++i)
        net.convolution.Kernels.data[i] = 0.03f * (static_cast<int>(i % 5) - 2);
    std::fill(net.convolution.Bias.data.begin(), net.convolution.Bias.data.end(), 0.6f);
    for (size_t index = 0; index < net.second_convolution.Kernels.data.size(); ++index)
        net.second_convolution.Kernels.data[index] = 0.07f * (static_cast<int>(index % 5) - 2);
    std::fill(net.second_convolution.Bias.data.begin(), net.second_convolution.Bias.data.end(), 0.5f);
    for (auto& layer : net.Layers) {
        for (size_t i = 0; i < layer.W.data.size(); ++i)
            layer.W.data[i] = 0.04f * (static_cast<int>(i % 7) - 3);
        std::fill(layer.b.data.begin(), layer.b.data.end(), 0.4f);
    }
}
std::vector<float*> parameters(NeuralNetwork& net) {
    std::vector<float*> result;
    for (float& value : net.convolution.Kernels.data) result.push_back(&value);
    for (float& value : net.convolution.Bias.data) result.push_back(&value);
    for (float& value : net.second_convolution.Kernels.data) result.push_back(&value);
    for (float& value : net.second_convolution.Bias.data) result.push_back(&value);
    for (auto& layer : net.Layers) {
        for (float& value : layer.W.data) result.push_back(&value);
        for (float& value : layer.b.data) result.push_back(&value);
    }
    return result;
}
Matrix<Layout::ColumnMajor> raw_batch(const std::vector<std::vector<float>>& raw,
                                     std::span<const size_t> indices) {
    Matrix<Layout::ColumnMajor> result(raw.front().size(), indices.size(), Initialization::Uninitialized);
    for (size_t index : indices)
        result.data.insert(result.data.end(), raw[index].begin(), raw[index].end());
    return result;
}

void check_gradients(ActivationType activation, LossType loss) {
    NeuralNetwork net({2, 4, 2}, 2, activation, loss, 4, 0.1f, 2, 2, 2);
    initialize(net);
    const auto raw = images();
    const auto transformed = mec(raw, 7, 8, 2);
    const std::vector<size_t> indices{2, 0, 2};
    const std::vector<int> labels{1, 0, 1};
    const auto input = raw_batch(raw, indices);
    auto params = parameters(net);
    std::vector<float> before, numerical;
    for (float* parameter : params) {
        const float saved = *parameter;
        before.push_back(saved);
        constexpr float epsilon = 0.002f;
        *parameter = saved + epsilon;
        const float plus = net.forward(transformed, indices, labels, Mode::test);
        *parameter = saved - epsilon;
        const float minus = net.forward(transformed, indices, labels, Mode::test);
        *parameter = saved;
        numerical.push_back((plus - minus) / (2 * epsilon * indices.size()));
    }
    net.forward(transformed, indices, labels, Mode::train);
    net.backward(input, labels);
    // Dense-Matrixoperatoren können den Speicher ersetzen; Zeiger danach neu abfragen.
    params = parameters(net);
    for (size_t i = 0; i < params.size(); ++i)
        close((before[i] - *params[i]) / net.learning_rate, numerical[i], "Netzwerkgradient");
    bool first_changed = false;
    for (size_t index = 0; index < net.convolution.Kernels.data.size(); ++index)
        first_changed = first_changed || std::abs(before[index] - *params[index]) > 1e-7f;
    require(first_changed, "Erste Convolution wurde nicht trainiert");
    const size_t second_start = net.convolution.Kernels.data.size() + net.convolution.Bias.data.size();
    bool second_changed = false;
    for (size_t index = second_start; index < second_start + net.second_convolution.Kernels.data.size(); ++index)
        second_changed = second_changed || std::abs(before[index] - *params[index]) > 1e-7f;
    require(second_changed, "Zweite Convolution wurde nicht trainiert");
    expect_throw<std::logic_error>([&] { net.backward(input, labels); });
}

void check_training_and_test() {
    NeuralNetwork trained({2, 4, 2}, 2, ActivationType::Sigmoid, LossType::CrossEntropy, 3, 0.1f, 2, 2, 2);
    NeuralNetwork manual({2, 4, 2}, 2, ActivationType::Sigmoid, LossType::CrossEntropy, 3, 0.1f, 2, 2, 2);
    initialize(trained);
    initialize(manual);
    const auto raw = images();
    const auto transformed = mec(raw, 7, 8, 2);
    const std::vector<int> labels{0, 1, 1, 0, 1};
    set_random_seed(123);
    trained.train(raw, transformed, labels, 5);
    require(trained.trained, "Trainingsstatus fehlt");
    set_random_seed(123);
    std::vector<size_t> order{0, 1, 2, 3, 4};
    std::shuffle(order.begin(), order.end(), random_generator());
    for (size_t start = 0; start < order.size(); start += 3) {
        const auto indices = std::span<const size_t>(order).subspan(start, std::min(size_t{3}, order.size() - start));
        std::vector<int> batch_labels;
        for (size_t index : indices) batch_labels.push_back(labels[index]);
        manual.forward(transformed, indices, batch_labels, Mode::train);
        manual.backward(raw_batch(raw, indices), batch_labels);
    }
    auto actual = parameters(trained), expected = parameters(manual);
    for (size_t i = 0; i < actual.size(); ++i)
        close(*actual[i], *expected[i], "Trainingsbatch", 1e-6f);

    std::vector<float> before;
    for (float* value : actual) before.push_back(*value);
    trained.test(transformed, labels);
    require(trained.Layers.back().a.columns == 2, "Test-Teilbatch fehlt");
    const std::vector<size_t> last{3, 4};
    manual.forward(transformed, last, {0, 1}, Mode::test_full);
    for (size_t i = 0; i < trained.Layers.back().a.data.size(); ++i)
        close(trained.Layers.back().a.data[i], manual.Layers.back().a.data[i], "Testausgabe", 1e-6f);
    actual = parameters(trained);
    for (size_t i = 0; i < actual.size(); ++i)
        close(*actual[i], before[i], "Test darf Gewichte nicht ändern", 1e-6f);
    expect_throw<std::logic_error>([&] { trained.backward(raw_batch(raw, last), {0, 1}); });

    // Auch Datensätze kleiner als batch_size und ein begrenztes Trainingsbudget funktionieren.
    const std::vector<std::vector<float>> single{raw[0]};
    trained.train(single, mec(single, 7, 8, 2), {0}, 1);
    require(trained.Layers.back().a.columns == 1, "Einzelbildbatch fehlt");
}

void check_pooling() {
    MaxPool2D pool;
    Matrix<Layout::ColumnMajor> gradient(4, 2, Initialization::Zero);
    expect_throw<std::logic_error>([&] { pool.backward(gradient); });
    Matrix<Layout::ColumnMajor> input(30, 2, Initialization::Zero);
    const std::vector<float> pixels{
        1,-4, 9,-2, 2,-8, 3,-7, 999,999,
        4,-3, 5,-2, 8,-6, 7,-9, 999,999,
        999,999, 999,999, 999,999, 999,999, 999,999};
    for (size_t b = 0; b < 2; ++b)
        for (size_t i = 0; i < 30; ++i) input(i,b) = pixels[i] + 10 * b;
    const auto output = pool.forward(input, 3, 5, 2);
    require(output.rows == 4 && output.columns == 2, "Pooling-Dimensionen");
    const std::vector<float> expected{9,-2,8,-6,19,8,18,4};
    for (size_t i = 0; i < expected.size(); ++i) {
        close(output.data[i], expected[i], "Pooling-Ausgabe", 0);
        gradient.data[i] = i + 1;
    }
    const auto backward = pool.backward(gradient);
    Matrix<Layout::ColumnMajor> expected_gradient(30, 2, Initialization::Zero);
    const std::vector<size_t> winners{2,3,14,15,32,33,44,45};
    for (size_t i = 0; i < winners.size(); ++i) expected_gradient.data[winners[i]] = i + 1;
    for (size_t i = 0; i < backward.data.size(); ++i)
        close(backward.data[i], expected_gradient.data[i], "Pooling-Gradient", 0);
    gradient.rows = 3;
    expect_throw<std::invalid_argument>([&] { pool.backward(gradient); });
    expect_throw<std::invalid_argument>([&] { pool.forward(input, 1, 15, 2); });
    require(MaxPool2D::output_size(26,26,8) == 1352, "MNIST-Pooling-Größe");
}

void check_pooling_reference() {
    MaxPool2D pool;
    for (size_t channels : {1u, 7u, 8u, 17u, 32u, 64u}) {
        for (size_t batch_size : {64u, 3u, 1u}) {
            const size_t rows = 27, columns = 26;
            Matrix<Layout::ColumnMajor> input(rows * columns * channels, batch_size, Initialization::Zero);
            for (size_t index = 0; index < input.data.size(); ++index)
                input.data[index] = static_cast<float>(static_cast<int>((index * 37 + index / 11) % 13) - 6);
            const auto output = pool.forward(input, rows, columns, channels);
            Matrix<Layout::ColumnMajor> gradient(output.rows, batch_size, Initialization::Zero);
            Matrix<Layout::ColumnMajor> expected(input.rows, batch_size, Initialization::Zero);
            size_t destination = 0;
            for (size_t batch = 0; batch < batch_size; ++batch)
                for (size_t row = 0; row < rows / 2; ++row)
                    for (size_t column = 0; column < columns / 2; ++column)
                        for (size_t channel = 0; channel < channels; ++channel) {
                            const size_t first = batch * input.rows + (2 * row * columns + 2 * column) * channels + channel;
                            size_t best = first;
                            for (size_t offset : {channels, columns * channels, (columns + 1) * channels})
                                if (input.data[first + offset] > input.data[best]) best = first + offset;
                            require(output.data[destination] == input.data[best], "Pooling-Referenzausgabe");
                            gradient.data[destination] = static_cast<float>(destination % 19 + 1);
                            expected.data[best] = gradient.data[destination++];
                        }
            require(pool.backward(gradient).data == expected.data, "Pooling-Referenzgradient inklusive Gleichstand");
        }
    }
}

void check_validation() {
    NeuralNetwork net({2, 4, 2}, 2, ActivationType::Sigmoid, LossType::CrossEntropy, 3, 0.1f, 2, 2, 2);
    const auto raw = images();
    auto transformed = mec(raw, 7, 8, 2);
    const std::vector<size_t> indices{0};
    expect_throw<std::invalid_argument>([&] { net.forward(transformed, indices, {2}, Mode::train); });
    expect_throw<std::invalid_argument>([&] { net.forward(transformed, indices, {}, Mode::train); });
    expect_throw<std::out_of_range>([&] { net.forward(transformed, std::vector<size_t>{5}, {0}, Mode::train); });
    net.forward(transformed, indices, {0}, Mode::train);
    Matrix<Layout::ColumnMajor> wrong(55, 1, Initialization::Zero);
    expect_throw<std::invalid_argument>([&] { net.backward(wrong, {0}); });
    transformed.data.pop_back();
    expect_throw<std::invalid_argument>([&] { net.forward(transformed, indices, {0}, Mode::train); });
    NeuralNetwork wrong_features({13, 4, 2}, 2, ActivationType::ReLU, LossType::CrossEntropy, 3, 0.1f, 2, 2, 2);
    expect_throw<std::invalid_argument>([&] { wrong_features.forward(mec(raw, 7, 8, 2), indices, {0}, Mode::train); });
    const std::vector<std::vector<float>> too_small{std::vector<float>(20)};
    NeuralNetwork large_kernel({2, 4, 2}, 2, ActivationType::ReLU, LossType::CrossEntropy, 3, 0.1f, 3, 2, 2);
    expect_throw<std::invalid_argument>([&] { large_kernel.forward(mec(too_small, 4, 5, 3), indices, {0}, Mode::train); });
    const std::vector<std::vector<float>> no_pooling_space{std::vector<float>(20)};
    expect_throw<std::invalid_argument>([&] { net.forward(mec(no_pooling_space, 4, 5, 2), indices, {0}, Mode::train); });
    const std::vector<std::vector<float>> no_first_pooling_space{std::vector<float>(6)};
    expect_throw<std::invalid_argument>([&] { net.forward(mec(no_first_pooling_space, 2, 3, 2), indices, {0}, Mode::train); });
    expect_throw<std::invalid_argument>([] {
        NeuralNetwork invalid({2, 4, 2}, 2, ActivationType::ReLU, LossType::CrossEntropy, 3, 0.1f, 2, 2, 0);
    });
}

void check_profiling() {
    const auto path = std::filesystem::temp_directory_path() /
        ("mnist-profiling-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".csv");
    {
        TimingProfiler profiler;
        int value = 42;
        int& disabled_reference = profiler.measure("disabled", [&]() -> int& { return value; });
        require(&disabled_reference == &value && profiler.statistics().empty(), "Deaktiviertes Profiling");
        profiler.enable(path.string());
        int& enabled_reference = profiler.measure("reference", [&]() -> int& { return value; });
        require(&enabled_reference == &value, "Profiling muss Referenzen erhalten");
        require(profiler.statistics().at("reference").calls == 1, "Profiler-Aufruf fehlt");

        NeuralNetwork measured({2, 4, 2}, 2, ActivationType::Sigmoid, LossType::CrossEntropy, 3, 0.1f, 2, 2, 2);
        NeuralNetwork plain({2, 4, 2}, 2, ActivationType::Sigmoid, LossType::CrossEntropy, 3, 0.1f, 2, 2, 2);
        initialize(measured);
        initialize(plain);
        measured.set_profiler(profiler);
        const auto raw = images();
        const auto transformed = mec(raw, 7, 8, 2);
        const std::vector<int> labels{0, 1, 1, 0, 1};
        set_random_seed(123);
        measured.train(raw, transformed, labels, 5);
        set_random_seed(123);
        plain.train(raw, transformed, labels, 5);
        const auto measured_parameters = parameters(measured), plain_parameters = parameters(plain);
        for (size_t index = 0; index < measured_parameters.size(); ++index)
            close(*measured_parameters[index], *plain_parameters[index], "Profiling verändert Gewichte", 0.0f);
        for (const auto& [name, stats] : profiler.statistics()) {
            require(stats.calls == 2, "Profiler muss Teilbatches mitzählen");
            require(std::isfinite(stats.total_ms) && stats.total_ms >= 0.0, "Ungültige Zeitmessung");
        }
        double stages_ms = 0.0;
        for (const auto& [name, stats] : profiler.statistics())
            if (!name.ends_with(".total")) stages_ms += stats.total_ms;
        require(stages_ms <= profiler.statistics().at("batch.total").total_ms, "Einzelstufen überlappen sich");
        require(profiler.statistics().at("backward.dense1.2x4").calls == 2, "Dense-Zeit fehlt");
        measured.test(transformed, labels);
        require(!profiler.statistics().contains("backward.total"), "Test enthält Trainingszeiten");
        require(profiler.statistics().at("forward.conv1_mec").calls == 2, "Test-Forward-Zeit fehlt");
    }
    {
        TimingProfiler appended;
        appended.enable(path.string());
        appended.measure("append", [] {});
        appended.report("setup", 0, 0, 0, false);
    }
    std::ifstream input(path);
    const std::string csv((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    require(csv.starts_with("run_id,phase,epoch,batches,samples,stage,kind,calls,total_ms,mean_ms,percent_batch\n"), "CSV-Header fehlt");
    require(csv.find("run_id", 1) == std::string::npos, "CSV-Header wurde erneut geschrieben");
    require(csv.find(",train,1,2,5,backward.dense1.2x4,stage,2,") != std::string::npos, "Trainings-CSV fehlt");
    require(csv.find(",test,0,2,5,forward.conv1_mec,stage,2,") != std::string::npos, "Test-CSV fehlt");
    require(csv.find(",setup,0,0,0,append,stage,1,") != std::string::npos, "CSV-Anhängen fehlt");
    input.close();
    std::filesystem::remove(path);
}

void check_mnist_architecture() {
    set_random_seed(163);
    NeuralNetwork net({800, 32, 16, 10}, 10, ActivationType::ReLU, LossType::CrossEntropy, 64, 0.025f, 3, 64);
    require(net.second_convolution.NUMBER_KERNELS == 32, "Zweite Convolution benötigt 32 Filter");
    require(net.second_convolution.INPUT_CHANNELS == 64, "Zweite Convolution benötigt 64 Eingabekanäle");
    require(net.second_convolution.Kernels.rows == 3 * 3 * 64, "Zweiter Kernel muss alle Kanäle abdecken");
    const std::vector<std::vector<float>> raw{std::vector<float>(28 * 28, 0.25f)};
    const std::vector<size_t> indices{0};
    const std::vector<int> labels{3};
    const auto transformed = mec(raw, 28, 28, 3);
    const float loss = net.forward(transformed, indices, labels, Mode::train);
    require(net.convolution.output().rows == 26 * 26 * 64, "Erste Convolution-Dimensionen");
    require(net.second_convolution.output().rows == 11 * 11 * 32, "Zweite Convolution nach erstem Pooling");
    require(net.Layers.front().W.columns == 5 * 5 * 32, "Dense benötigt 800 Features nach zweitem Pooling");
    require(std::isfinite(loss), "MNIST-Loss muss endlich sein");
    float probability_sum = 0.0f;
    for (float probability : net.Layers.back().a.data) probability_sum += probability;
    close(probability_sum, 1.0f, "MNIST-Wahrscheinlichkeiten");
    const auto first_before = net.convolution.Kernels.data;
    const auto second_before = net.second_convolution.Kernels.data;
    net.backward(raw_batch(raw, indices), labels);
    require(first_before != net.convolution.Kernels.data, "Erste MNIST-Convolution wurde nicht trainiert");
    require(second_before != net.second_convolution.Kernels.data, "Zweite MNIST-Convolution wurde nicht trainiert");
    for (float* parameter : parameters(net))
        require(std::isfinite(*parameter), "MNIST-Parameter müssen endlich sein");
    require(std::isfinite(net.forward(transformed, indices, labels, Mode::test)), "MNIST-Ausgabe nach Training");
}
}
int main() {
    for (auto activation : {ActivationType::Sigmoid, ActivationType::ReLU})
        for (auto loss : {LossType::CrossEntropy, LossType::MeanSquareError})
            check_gradients(activation, loss);
    check_pooling();
    check_pooling_reference();
    check_training_and_test();
    check_validation();
    check_mnist_architecture();
    check_profiling();
    std::cout << "Alle Netzwerk-Convolution-Tests erfolgreich.\n";
}
