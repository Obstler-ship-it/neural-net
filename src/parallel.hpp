#pragma once

#include <algorithm>
#include <cstdlib>
#include <exception>
#include <stdexcept>
#include <thread>
#include <vector>

namespace parallel {
inline size_t worker_count(size_t items, size_t work_per_item) {
    if (items < 2 || work_per_item * items < 1000000) return 1;
    static const size_t limit = [] {
        if (const char* setting = std::getenv("MNIST_THREADS")) {
            char* end = nullptr;
            const long value = std::strtol(setting, &end, 10);
            if (end == setting || *end != '\0' || value < 1 || value > 256)
                throw std::invalid_argument("MNIST_THREADS must be between 1 and 256");
            return static_cast<size_t>(value);
        }
        return static_cast<size_t>(std::max(1u, std::min(8u, std::thread::hardware_concurrency())));
    }();
    return std::min(items, limit);
}

template<typename Function>
void for_ranges(size_t items, size_t workers, Function function) {
    if (workers == 1) {
        function(0, 0, items);
        return;
    }
    std::vector<std::exception_ptr> errors(workers);
    auto run = [&](size_t worker) {
        try {
            function(worker, items * worker / workers, items * (worker + 1) / workers);
        } catch (...) {
            errors[worker] = std::current_exception();
        }
    };
    std::vector<std::jthread> threads;
    threads.reserve(workers - 1);
    for (size_t worker = 1; worker < workers; ++worker) threads.emplace_back(run, worker);
    run(0);
    for (auto& thread : threads) thread.join();
    for (const auto& error : errors)
        if (error) std::rethrow_exception(error);
}
}
