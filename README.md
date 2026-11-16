# MNIST Neural Network in C++

A small neural network that recognizes handwritten digits from the [MNIST dataset](https://learn.microsoft.com/en-us/azure/open-datasets/dataset-mnist). It is written from scratch in C++ using only stdlib and includes matrix operations, dense layers, activation functions, optimizers, validation, and a training loop.

The current example builds a dense network with ReLU activations and Momentum, trains on a shuffled 90% training split, uses the remaining 10% for validation, and reports accuracy on MNIST's separate test set.

## Requirements

- CMake 3.16 or newer
- A C++23 compiler and standard library with `<print>` and `std::println`
- `curl` and `gzip` to download the dataset

The project has been built with Clang 18 and libstdc++ 14. On Ubuntu, install the tools with:

```bash
sudo apt install clang g++ cmake curl gzip
clang++ --version
cmake --version
```

## Dataset

Place the four uncompressed MNIST files in `data/`:

```text
data/train-images.idx3-ubyte
data/train-labels.idx1-ubyte
data/t10k-images.idx3-ubyte
data/t10k-labels.idx1-ubyte
```

To download them from Microsoft's mirror:

```bash
mkdir -p data
curl -fL -o data/train-images.idx3-ubyte.gz https://azureopendatastorage.blob.core.windows.net/mnist/train-images-idx3-ubyte.gz
curl -fL -o data/train-labels.idx1-ubyte.gz https://azureopendatastorage.blob.core.windows.net/mnist/train-labels-idx1-ubyte.gz
curl -fL -o data/t10k-images.idx3-ubyte.gz https://azureopendatastorage.blob.core.windows.net/mnist/t10k-images-idx3-ubyte.gz
curl -fL -o data/t10k-labels.idx1-ubyte.gz https://azureopendatastorage.blob.core.windows.net/mnist/t10k-labels-idx1-ubyte.gz
gzip -dk data/*-ubyte.gz
```

## Build and run

For the `first` time starting you will need the following instructions.
Run these commands from the project directory:

```bash
cmake -S . -B build-clang -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_BUILD_TYPE=Release
```

afterwards you just need to type this command to start.
`start.sh` rebuilds the program and runs it.

```bash
bash start.sh
```

## Network API

Choose an optimizer as the network's template argument, provide the input width and batch size, then add dense layers and activations in order:

```cpp
NeuralNetwork<Momentum> net(784, 64);
net.add<Dense>(128);
net.add<ReLU>();
net.add<Dense>(32);
net.add<ReLU>();
net.add<Dense>(10);
net.set_loss<CrossEntropy>();

auto [train, validation] = split_dataset(get_training_data());
net.train(train.images, train.labels,
          validation.images, validation.labels);

const auto [test_images, test_labels] = get_test_data();
net.test(test_images, test_labels);
```

Available optimizers are `Adam`, `SGD`, `Momentum`, and `RMSprop`. Available activations are `ReLU` and `Sigmoid`; available losses are `CrossEntropy` and `MeanSquareError`. Softmax is applied automatically at the output. The optimizer uses its default configuration in the current network API.

The older constructor taking a list of layer widths, `ActivationType`, and `LossType` remains available.

## TODO

-Implementing some kind of print for the Digits in the MNIST Dataset.
