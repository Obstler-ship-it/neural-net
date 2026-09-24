# MNIST Neural Network in C++

This project trains a small neural network to recognize handwritten digits from the [MNIST dataset](https://learn.microsoft.com/en-us/azure/open-datasets/dataset-mnist). It implements its own matrix operations, layers, activation functions, and training loop. The program trains on 60,000 images and then reports its accuracy on 10,000 test images.

In a local run, the network reached **98,07% accuracy** on the MNIST test set. With NeuralNetwork net( {784, 256, 256, 128, 32, 10}, ActivationType::ReLU, LossType::CrossEntropy );.

The instructions below are for Linux or Windows Subsystem for Linux (WSL). Run every command from the project directory: the directory containing `CMakeLists.txt`.

## What you need

- CMake 3.16 or newer.
- A C++23 compiler **and standard library** that provide `<print>` and `std::println`. This project has been built with Clang 18 using libstdc++ 14. GCC 13's standard library does not provide `<print>`.
- `curl` and `gzip` if you need to download the dataset.

Check that the tools are available:

```bash
cmake --version
clang++ --version
```

## 1. Get the dataset

The program expects these four **uncompressed** files in the project's `data/` directory:

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

The names of the downloaded `.gz` files intentionally use dots: after decompression, they match the names expected by the program.

## 2. Configure and compile

From the project directory, run:

```bash
cmake -S . -B build-clang -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_BUILD_TYPE=Release
cmake --build build-clang --parallel
```

The first command configures the build; the second compiles the executable at `build-clang/MNIST`. You only need to configure once for this build directory. After changing source code, run the second command again.

## 3. Run

```bash
./build-clang/MNIST
```

The program prints timing and average loss for each epoch, followed by test accuracy. For example, `Genauigkeit: 0.9755` means 97.55% of the test images were classified correctly. Results and runtime can vary with hardware and code changes.

You can also use `bash start.sh` **after step 2**. The script rebuilds `build-clang` and starts its executable; it does not download the data or configure a new build directory.

## Common problems

- **`CMakeLists.txt` not found:** Run the `cmake -S . ...` command from the project directory, not from inside `build/` or `build-clang/`.
- **`fatal error: 'print' file not found`:** The compiler's standard library lacks C++23 `<print>`. Use a toolchain that provides it. CMake remembers the compiler selected for each build directory, so configure a new build directory after changing compilers.
- **`Could not open MNIST files`:** Check the four uncompressed filenames in `data/`. `.gz` archives alone are not enough.
- **Old output after editing code:** Rebuild with `cmake --build build-clang --parallel` and run `./build-clang/MNIST`. `./MNIST` inside `build/` starts a different executable.
- **`./start.sh: Permission denied`:** Run `bash start.sh`, or grant execute permission once with `chmod +x start.sh`.

The Release build enables `-O3` and `-ffast-math`. The latter can make checks for `NaN` and infinity unreliable; use a build without that flag when investigating numerical errors.
