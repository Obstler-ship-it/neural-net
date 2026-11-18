CXX = clang++
CXXFLAGS = -std=c++23 -Isrc

ifeq ($(DEBUG),1)
CXXFLAGS += -g -O0
endif

TARGET = build/matrix_test
SOURCE = src/tests/matrix_test.cpp
MATRIX_SOURCE = src/matrix/matrix.hpp src/matrix/matrix.cpp
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

.PHONY: adam-test
adam-test: build/adam_test
	./build/adam_test

build/adam_test: src/tests/adam_test.cpp src/optimizers/adam.hpp $(MATRIX_SOURCE)
	mkdir -p build
	$(CXX) $(CXXFLAGS) -O2 $< -o $@

.PHONY: loss-test
loss-test: build/loss_test
	./build/loss_test

build/loss_test: src/tests/loss_test.cpp src/loss/mse.hpp src/loss/loss_function.hpp src/loss/cross_entropy.hpp src/softmax/softmax.hpp $(MATRIX_SOURCE)
	mkdir -p build
	$(CXX) $(CXXFLAGS) $< -o $@
