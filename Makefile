CXX = clang++
CXXFLAGS = -std=c++23 -Isrc

ifeq ($(DEBUG),1)
CXXFLAGS += -g -O0
endif

TARGET = build/matrix_test
SOURCE = src/tests/matrix_test.cpp
MATRIX_SOURCE = src/matrix/matrix.cpp
BENCHMARK_TARGET = build/matrix_benchmark
BENCHMARK_SOURCE = src/tests/matrix_benchmark.cpp
BENCHMARK_FLAGS = -O3 -march=native

$(TARGET): $(SOURCE) $(MATRIX_SOURCE)
	mkdir -p build
	$(CXX) $(CXXFLAGS) -include $(MATRIX_SOURCE) $(SOURCE) -o $(TARGET)

benchmark: $(BENCHMARK_TARGET)

$(BENCHMARK_TARGET): $(BENCHMARK_SOURCE) $(MATRIX_SOURCE)
	mkdir -p build
	$(CXX) $(CXXFLAGS) $(BENCHMARK_FLAGS) -include $(MATRIX_SOURCE) $(BENCHMARK_SOURCE) -o $(BENCHMARK_TARGET)

clean:
	rm -rf build
