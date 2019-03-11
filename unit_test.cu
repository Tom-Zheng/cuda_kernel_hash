#include <stdio.h>
#include "kernel_hash.cuh"
#include <array>
#include <random>
#include <iostream>
#include <algorithm>
#include <unordered_map>

__global__ void get_index(RH_hash_table<uint32_t, uint32_t> * d_hashtables, uint32_t* input_keys, uint32_t* input_vals){
    if(blockIdx.x < 1024)   {
        d_hashtables->lock();
        d_hashtables->insert(input_keys[blockIdx.x], input_vals[blockIdx.x]);
        d_hashtables->unlock();
    }
}
    
void unit_test(RH_hash_table<uint32_t, uint32_t>* h_hashtables) {
    // TODO
    // Generate test data
    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_int_distribution<std::mt19937::result_type> dist(0, 127); // distribution in range [1, 6]
    
    std::array<uint32_t, 1024> numpool;
    std::array<uint32_t, 1024> valpool;
    std::unordered_map<uint32_t, uint32_t> cpu_hash;
    
    // Generate number pool
    for(int i = 0; i < numpool.size(); i++) {
        numpool[i] = i;
        valpool[i] = dist(rng);
    }
    
    std::shuffle(numpool.begin(), numpool.end(), rng);

    cudaEvent_t start, stop;
    // Stopwatch
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    // Generate CPU answer
    for(int i = 0; i < numpool.size(); i++) {
        cpu_hash[numpool[i]] = valpool[i];
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf( "CPU Time elapsed: %3.1f ms\n", elapsedTime);

    // GPU
    uint32_t* d_numpool = NULL;
    uint32_t* d_valpool = NULL;
    gpuErrchk(cudaMalloc((void**)&d_numpool, sizeof(uint32_t) * numpool.size()));
    gpuErrchk(cudaMemcpy(d_numpool, numpool.data(), sizeof(uint32_t) * numpool.size(), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMalloc((void**)&d_valpool, sizeof(uint32_t) * numpool.size()));
    gpuErrchk(cudaMemcpy(d_valpool, valpool.data(), sizeof(uint32_t) * valpool.size(), cudaMemcpyHostToDevice));

    RH_hash_table<uint32_t, uint32_t> * d_hashtables = NULL;  
    gpuErrchk(cudaMalloc((void**)&d_hashtables, sizeof(RH_hash_table<uint32_t, uint32_t>)));
    gpuErrchk(cudaMemcpy(d_hashtables, h_hashtables, sizeof(RH_hash_table<uint32_t, uint32_t>), cudaMemcpyHostToDevice));

    cudaEventRecord(start, 0);
    // Does Actual work
    get_index<<<1024,1>>>(d_hashtables, d_numpool, d_valpool);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf( "GPU Time elapsed: %3.1f ms\n", elapsedTime);
    
    uint32_t *h_keys = new uint32_t[h_hashtables->capacity];
    uint32_t *h_vals = new uint32_t[h_hashtables->capacity];
    uint32_t *h_hash = new uint32_t[h_hashtables->capacity];

    gpuErrchk(cudaMemcpy(h_keys, h_hashtables->buffer_keys, sizeof(uint32_t) * h_hashtables->capacity, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(h_vals, h_hashtables->buffer_values, sizeof(uint32_t) * h_hashtables->capacity, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(h_hash, h_hashtables->buffer_hash, sizeof(uint32_t) * h_hashtables->capacity, cudaMemcpyDeviceToHost));

    for(int i = 0; i < h_hashtables->capacity; i++)  {
        if(h_hash[i] != 0)  {
            if(cpu_hash[h_keys[i]] != h_vals[i])    {
                printf("key = %d, val = %d\n", h_keys[i], h_vals[i]);
                fprintf(stderr,"Value does not match\n");    
            }
        }
    }
    delete [] h_keys;
    delete [] h_vals;
    delete [] h_hash;
}

int main() {
    RH_hash_table<uint32_t, uint32_t> h_hashtable;

    unit_test(&h_hashtable);

    return 0;
}
