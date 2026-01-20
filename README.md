# MNIST Neural Network in C

A Multilayer Perceptron (MLP) implementation written in pure C to classify MNIST handwritten digits.
No external machine learning libs

## Architecture
- Input: 784 nodes (28x28 pixels)
- Hidden: 128 nodes (ReLU activation)
- Output: 10 nodes (Sigmoid activation)

## Setup and Compilation

1. Download the dataset:
   chmod +x download.sh
   ./download.sh

2. Compile the source code:
   gcc -O3 -march=native -Wno-unused-result src/main.c src/nn.c src/mnist.c -lm

3. Run the application:
   ./a.out
