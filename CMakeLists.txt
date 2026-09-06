cmake_minimum_required(VERSION 3.16)

project(MNIST LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(MNIST
    main.cpp
    src/mnist_reader/reader.cpp
    src/mnist_reader/reader.hpp
)

target_include_directories(MNIST PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
