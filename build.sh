#!/bin/bash
mkdir -p build
rm build/*
nvcc -include kernel_hash.cuh --std=c++11 unit_test.cu -o build/unit_test
./build/unit_test