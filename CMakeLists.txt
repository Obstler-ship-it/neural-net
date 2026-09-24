cmake_minimum_required(VERSION 3.16)

project(MNIST LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(MNIST
    main.cpp
    mnist_reader/reader.cpp
    src/layer/layer.cpp
    src/neural_network.cpp
)

target_include_directories(MNIST PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_compile_definitions(MNIST PRIVATE
    MNIST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(MNIST PRIVATE
        $<$<CONFIG:Release>:-O3>
        $<$<CONFIG:Release>:-ffast-math>
    )
endif()
