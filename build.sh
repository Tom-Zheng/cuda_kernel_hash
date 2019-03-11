#!/bin/bash
mkdir -p build
rm build/*
nvcc --std=c++11 kernel_hash.cu -o build/kernel_hash
./build/kernel_hash