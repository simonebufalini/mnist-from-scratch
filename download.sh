#!/bin/bash

set -e
mkdir -p dataset/training
mkdir -p dataset/testing

# URL Mirror
BASE_URL="https://storage.googleapis.com/cvdf-datasets/mnist"

wget -c "${BASE_URL}/train-images-idx3-ubyte.gz" -O dataset/training/train-images-idx3-ubyte.gz
wget -c "${BASE_URL}/train-labels-idx1-ubyte.gz" -O dataset/training/train-labels-idx1-ubyte.gz

wget -c "${BASE_URL}/t10k-images-idx3-ubyte.gz" -O dataset/testing/t10k-images-idx3-ubyte.gz
wget -c "${BASE_URL}/t10k-labels-idx1-ubyte.gz" -O dataset/testing/t10k-labels-idx1-ubyte.gz

echo "Exctracting..."
# exctract keeping original file
gunzip -k -f dataset/training/*.gz
gunzip -k -f dataset/testing/*.gz

echo "Dataset in ./dataset/"
