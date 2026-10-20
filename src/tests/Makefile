CXX = clang++
CXXFLAGS = -std=c++23 -Isrc

ifeq ($(DEBUG),1)
CXXFLAGS += -g -O0
endif

TARGET = build/matrix_test
SOURCE = src/tests/matrix_test.cpp
MATRIX_SOURCE = src/matrix/matrix.hpp src/matrix/matrix.tpp
BENCHMARK_TARGET = build/matrix_benchmark
BENCHMARK_SOURCE = src/tests/matrix_benchmark.cpp
BENCHMARK_FLAGS = -O3 -march=native

$(TARGET): $(SOURCE) $(MATRIX_SOURCE)
	mkdir -p build
	$(CXX) $(CXXFLAGS) $(SOURCE) -o $(TARGET)

benchmark: $(BENCHMARK_TARGET)

$(BENCHMARK_TARGET): $(BENCHMARK_SOURCE) $(MATRIX_SOURCE)
	mkdir -p build
	$(CXX) $(CXXFLAGS) $(BENCHMARK_FLAGS) $(BENCHMARK_SOURCE) -o $(BENCHMARK_TARGET)

clean:
	rm -rf build

.PHONY: convolution-test
convolution-test: build/convolution_test
	./build/convolution_test

build/convolution_test: src/tests/convolution_test.cpp src/convolution/convolution.cpp $(wildcard src/convolution/*.hpp src/activation/*.hpp src/mec/*.hpp) $(MATRIX_SOURCE)
	mkdir -p build
	$(CXX) $(CXXFLAGS) src/tests/convolution_test.cpp src/convolution/convolution.cpp -o $@

NETWORK_TEST_SOURCES = src/tests/neural_network_test.cpp src/neural_network.cpp src/layer/layer.cpp src/convolution/convolution.cpp
.PHONY: network-test
network-test: build/neural_network_test
	./build/neural_network_test

build/neural_network_test: $(NETWORK_TEST_SOURCES) $(wildcard src/*.hpp src/*/*.hpp src/matrix/*.tpp)
	mkdir -p build
	$(CXX) $(CXXFLAGS) $(NETWORK_TEST_SOURCES) -o $@
