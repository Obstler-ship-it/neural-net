#!/usr/bin/env bash
set -e
cmake --build build-clang
cd build-clang
./MNIST
