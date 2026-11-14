# MNIST Neural Network in C++

This project trains a small neural network to recognize handwritten digits from the [MNIST dataset](https://learn.microsoft.com/en-us/azure/open-datasets/dataset-mnist). It implements its own matrix operations, layers, activation functions, convolution and training loop. The program trains on 60,000 images and then reports its accuracy on 10,000 test images. The neural network is implemented from scratch in C++ using only the standard library.

In a local run, the network reached **98,07% accuracy** on the MNIST test set. With NeuralNetwork net( {784, 256, 256, 128, 32, 10}, ActivationType::ReLU, LossType::CrossEntropy );.

The instructions below are for **Linux** or Windows Subsystem for Linux (WSL). Run every command from the project directory: the directory containing `CMakeLists.txt`.

## Network architecture

The current configuration in `main.cpp` uses two trainable convolutions with ReLU, stride 1 and no padding:

```text
28 x 28 x 1 input
  -> 3 x 3 convolution, 64 filters -> 26 x 26 x 64
  -> ReLU -> 2 x 2 max-pooling    -> 13 x 13 x 64
  -> 3 x 3 convolution, 32 filters -> 11 x 11 x 32
  -> ReLU -> 2 x 2 max-pooling    -> 5 x 5 x 32
  -> flatten                     -> 800 features
  -> dense 128 (ReLU) -> dense 32 (ReLU) -> dense 16 (ReLU) -> dense 10 -> softmax
```

Each filter in the second convolution spans all 64 input channels. Backpropagation passes through both pooling stages and updates both convolution layers. The final `NeuralNetwork` constructor argument controls the second layer's filter count (default: 32); `layer_sizes[0]` must match the flattened size after both convolution-pooling blocks. Both convolutions use the specified `kernel_size`. Both pooling stages use stride 2 and discard an unmatched final row or column. In particular, the second pooling maps 11 x 11 to 5 x 5.

Pooling before the second convolution reduces its spatial output from 24 x 24 to 11 x 11: about 4.76 times fewer convolution multiplications. The first dense layer receives 800 rather than 4608 features. Actual batch speedups also depend on the first convolution, pooling, memory access and threading overhead.

The first convolution consumes precomputed MEC data. The second consumes the first layer's pooled feature maps directly, stored as `(row * width + column) * channels + channel`, with one image per matrix column. Its weights use `(kernel_column * kernel_size + kernel_row) * input_channels + channel` as the matrix row and the output filter as the matrix column.

## CPU performance

Convolutions use small matrix blocks directly over image rows, avoiding large intermediate patch matrices. Release builds use an AVX2/FMA kernel when those instructions are available at compile time, with a portable C++ fallback otherwise. Images in a batch are distributed across up to eight CPU threads; each thread accumulates its own weight gradients before a final reduction. No external BLAS or machine-learning library is required.

Convolution outputs reuse internal storage. `Convolution::forward()` and `output()` return a const reference that is overwritten by the next forward call on that layer. Use a value copy if an output must be retained longer. The network consumes these references directly to avoid copying feature maps.

The default thread count is the smaller of eight and the hardware concurrency. Override it with `MNIST_THREADS`, for example:

```bash
MNIST_THREADS=4 bash start.sh
```

For a repeatable comparison, build and run the synthetic batch benchmark:

```bash
make -f src/tests/Makefile network-benchmark
./build/network_benchmark 20
MNIST_THREADS=1 ./build/network_benchmark 20
```

It measures the existing 64-filter/32-filter network with batches of 64, two warmup iterations and median times for the requested number of iterations. Dataset loading and MEC preparation are excluded. Compare builds on the same machine without running other builds or training jobs concurrently. Thread count and SIMD change floating-point summation order, so training results can differ slightly.

The benchmark also reports isolated Pool1 forward/backward times, including output allocation. Pooling writes directly into a sized output buffer and tracks the winning value alongside its index, avoiding dependent reloads. Equal values still select the first position. Both pooling layers share this implementation; pooling remains single-threaded.

## Detailed timing diagnostics

Profiling is opt-in and does not change the model, optimizer or training budget:

```bash
MNIST_PROFILE=profiling.csv bash start.sh
```

`start.sh` changes into `build-clang`, so this writes `build-clang/profiling.csv`. An absolute path also works. Results are appended with a new `run_id`; existing runs are retained. For a short diagnostic run that trains on 6400 samples and then evaluates the full test set:

```bash
MNIST_PROFILE=profiling-short.csv MNIST_TRAIN_SAMPLES=6400 bash start.sh
```

Without `MNIST_TRAIN_SAMPLES`, the existing default training budget is used. The override counts processed training samples, including repeats across epochs; it does not resize the loaded dataset. Accuracy after a short diagnostic run is not a full-training accuracy measurement.

Console tables and CSV snapshots are emitted every 100 batches and at each epoch/test end. Times are cumulative within the current epoch or test phase; use the last snapshot for that phase rather than summing snapshots. Setup timing separately records dataset loading, network initialization and MEC conversion. The training setup also prints raw pixel bytes, used MEC bytes and reserved MEC capacity.

Every Conv, Pool and Dense layer has separate forward/backward rows. Dense names include the input and output dimensions, for example `forward.dense1.800x128`. Convolution timings include activation, memory work, thread startup, waiting and gradient reduction. Dense backward includes weight gradients, input gradients and parameter updates. Pooling outputs already have flattened storage, so there is no separate flatten operation to time. These measurements identify expensive stages; they do not separate arithmetic from memory stalls within a stage.

CSV columns: `run_id,phase,epoch,batches,samples,stage,kind,calls,total_ms,mean_ms,percent_batch`. All durations are elapsed wall-clock milliseconds, not summed CPU time across threads. `batch.total`, `forward.total` and `backward.total` are **inclusive totals**: do not add them to their constituent stages. `percent_batch` uses the cumulative `batch.total` as denominator; it is blank for setup measurements. Small gaps include validation, loop bookkeeping and instrumentation overhead. Batch timing excludes console/CSV output, epoch shuffling and dataset setup; the existing overall test timer includes reporting overhead. Initial batches and partial batches are included, so compare equivalent runs with the same thread count.

To judge MEC itself, `make -f src/tests/Makefile network-benchmark` followed by `./build/network_benchmark 20` now measures Conv1 on identical inputs both with precomputed MEC and directly from raw pixels. MEC preparation time and memory are reported separately. This is a comparison of the two implemented convolution paths, not a comparison against an external optimized library. Conv2 uses pooled feature maps and does not use MEC preprocessing.

## What you need

- CMake 3.16 or newer.
- A C++23 compiler **and standard library** that provide `<print>` and `std::println`. This project has been built with Clang 18 using libstdc++ 14. GCC 13's standard library does not provide `<print>`.
- `curl` and `gzip` if you need to download the dataset.

Check that the tools are available:

```bash
sudo apt install clang g++ cmake curl gzip
clang++ --version
cmake --version
```

## 1. Get the dataset

The program expects these four **uncompressed** files in the project's `neural-net/data/` directory:

```text
data/train-images.idx3-ubyte
data/train-labels.idx1-ubyte
data/t10k-images.idx3-ubyte
data/t10k-labels.idx1-ubyte
```

If they are already there, skip to step 2. Otherwise, download the MNIST archives from [Microsoft's MNIST dataset mirror](https://learn.microsoft.com/en-us/azure/open-datasets/dataset-mnist) and unpack them:

```bash
mkdir -p data
curl -fL -o data/train-images.idx3-ubyte.gz https://azureopendatastorage.blob.core.windows.net/mnist/train-images-idx3-ubyte.gz
curl -fL -o data/train-labels.idx1-ubyte.gz https://azureopendatastorage.blob.core.windows.net/mnist/train-labels-idx1-ubyte.gz
curl -fL -o data/t10k-images.idx3-ubyte.gz https://azureopendatastorage.blob.core.windows.net/mnist/t10k-images-idx3-ubyte.gz
curl -fL -o data/t10k-labels.idx1-ubyte.gz https://azureopendatastorage.blob.core.windows.net/mnist/t10k-labels-idx1-ubyte.gz
gzip -dk data/*-ubyte.gz
```

## 2. Run

For the **first** time starting you will need the following instructions

```bash
cmake -S . -B build-clang -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_BUILD_TYPE=Release
chmod +x start.sh
```

afterwards you just need to type this command to start

```bash
bash start.sh
```

## Tests

```bash
make -B -f src/tests/Makefile gemm-test convolution-test network-test
make -B -f src/tests/Makefile OPTIMIZED=1 gemm-test convolution-test network-test
```

These tests check convolution outputs and numerical gradients, including multiple input channels, both convolution layers, pooling and partial batches. They also compare blocked matrix products against a scalar reference (including incomplete SIMD blocks) and batched gradients against individual-image gradients. The second command exercises the native SIMD path; test builds deliberately retain finite-value checks rather than enabling `-ffast-math`.

#TODO

-Algo for estimating the learning-rate and to stop the training at some point
-Functionality / Print for one random Digit and the probabilities
-start via consol with std::cin
