#pragma once

#include <random>

inline std::mt19937& random_generator() {
    static std::mt19937 generator{42};
    return generator;
}

inline void set_random_seed(unsigned int seed) {
    random_generator().seed(seed);
}
