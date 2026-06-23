#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <algorithm>
#include <cfloat>

#include "../../kernels/softmax_online.cuh"

#define SIZE 32768
#define BLOCKSIZE 1024

// array intialization
void arr_init(float *arr_in, float * arr_out, int size) {
    for (int i = 0; i < size; ++i) {
        arr_in[i] = (float)(rand() % 100);
        arr_out[i] = 0;
    }
}

// sum checker
void sum_check(float *cpu, float *gpu) {
    float cpu_sum = 0;
    float gpu_sum = 0;

    for (int i = 0; i < SIZE; ++i) {
        cpu_sum += cpu[i];
    }

    for (int i = 0; i < SIZE / (BLOCKSIZE * 2); ++i) {
        gpu_sum += gpu[i];
    }

    if (fabs((gpu_sum - cpu_sum) / cpu_sum) < 1e-3f) {
        std::cout << "Correct!: " << gpu_sum << " = " << cpu_sum << std::endl;
    }

    else {
        std::cout << "Wrong! GPU Sum: " << gpu_sum << " CPU Sum: " << cpu_sum << " Diff: " << gpu_sum - cpu_sum << std::endl;
    }
}

void max_check(float *cpu, float *gpu) {
    float cpu_max = *std::max_element(cpu, cpu + SIZE);
    float gpu_max = *gpu;

    if (cpu_max == gpu_max) {
        std::cout << "Correct: GPU Max: " << gpu_max << " CPU Max: " << cpu_max << std::endl;
    }

    else {
        std::cout << "Wrong! GPU Max: " << gpu_max << " CPU Max: " << cpu_max << " Diff: " << gpu_max - cpu_max << std::endl;
    }
}

// 
void softmax_check(float *cpu_in, float *gpu_sm) {
    float cpu_max = *std::max_element(cpu_in, cpu_in + SIZE);
    float cpu_sum = 0;
    float *cpu_sm = (float*)malloc(SIZE * sizeof(float));

    for (int i = 0; i < SIZE; ++i) {
        cpu_sum += exp(cpu_in[i] - cpu_max);
    }

    for (int i = 0; i < SIZE; ++i) {
        cpu_sm[i] = std::exp(cpu_in[i] - cpu_max) / cpu_sum;
    }

    bool correct = true;
    int idx = 0;

    for (int i = 0; i < SIZE; ++i) {
        if (fabs((gpu_sm[i] - cpu_sm[i]) / cpu_sm[i]) > 1e-3f) {
            correct = false;
            idx = i;
            break;
        }
    }

    if (!correct) {
        std::cout << "Incorrect: " << gpu_sm[idx] << " CPU Max: " << cpu_sm[idx] << " Diff: " << gpu_sm[idx] - cpu_sm[idx] << std::endl;
    }
    else {
        std::cout << "Correct!" << std::endl;
    } 

    free(cpu_sm);
}

int main() {
    int size = SIZE;

    // mem sizes
    size_t in_bytes = size * sizeof(float);
    size_t out_bytes = size * sizeof(float);

    // host memory allocation
    float *h_in, *h_out;

    h_in = (float*)malloc(in_bytes);
    h_out = (float*)malloc(out_bytes);

    // array initialization
    arr_init(h_in, h_out, size);

    // device memory allocation
    float *d_in, *d_out;

    cudaMalloc(&d_in, in_bytes);
    cudaMalloc(&d_out, out_bytes);

    // copy data to GPU
    cudaMemcpy(d_in, h_in, in_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_out, h_out, out_bytes, cudaMemcpyHostToDevice);

    // grid and block dims (one block does entire arr)
    dim3 gridDim(1, 1, 1);
    dim3 blockDim(BLOCKSIZE);

    // launch partial max kernel and fetch results
    softmax<<<gridDim, blockDim>>>(d_in, d_out);
    cudaDeviceSynchronize();

    cudaMemcpy(h_out, d_out, out_bytes, cudaMemcpyDeviceToHost);
    softmax_check(h_in, h_out);

    // free memory
    free(h_in);
    free(h_out);
    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}