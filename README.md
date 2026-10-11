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

```bash
chmod +x start.sh
bash start.sh
```

#TODO

Implementing Convolution Layers
Algo for estimating the learning-rate and to stop the training at some point
Functionality / Print for one random Digit and the probabilities
start via consol with std::cin
