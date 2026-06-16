#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <algorithm>

#include "../kernels/sum_naive.cuh" 
#include "../kernels/sum_shuffle.cuh" 
#include "../kernels/max_reduce.cuh" 
#include "../kernels/softmax_two_pass.cuh" 

#define SIZE 32768
#define BLOCKSIZE 128

// array intialization
void arr_init(float *arr_in, float *arr_partial_max, float *full_max, float *arr_exp, float *arr_partial_sum, float *full_sum, float * arr_out, int size) {
    for (int i = 0; i < size; ++i) {
        arr_in[i] = (float)(rand() % 100);
        arr_partial_max[i] = 0;
        arr_exp[i] = 0;
        arr_partial_sum[i] = 0;
        arr_out[i] = 0;
    }

    *full_max = 0;
    *full_sum = 0;
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
    // arr size
    int size = SIZE;

    // mem sizes
    size_t in_bytes = size * sizeof(float);
    size_t partial_max_bytes = size * sizeof(float);
    size_t full_max_bytes = sizeof(float);
    size_t exp_bytes = size * sizeof(float);
    size_t partial_sum_bytes = size * sizeof(float);
    size_t full_sum_bytes = sizeof(float);
    size_t out_bytes = size * sizeof(float);

    // host memory allocation
    float *h_in, *h_partial_max, *h_full_max, *h_exp, *h_partial_sum, *h_full_sum, *h_out;

    h_in = (float*)malloc(in_bytes);
    h_partial_max = (float*)malloc(partial_max_bytes);
    h_full_max = (float*)malloc(full_max_bytes);
    h_exp = (float*)malloc(exp_bytes);
    h_partial_sum = (float*)malloc(partial_sum_bytes);
    h_full_sum = (float*)malloc(full_sum_bytes);
    h_out = (float*)malloc(out_bytes);

    // array initialization
    arr_init(h_in, h_partial_max, h_full_max, h_exp, h_partial_sum, h_full_sum, h_out, size);

    // device memory allocation (GPU)
    float *d_in, *d_partial_max, *d_full_max, *d_exp, *d_partial_sum, *d_full_sum, *d_out;

    cudaMalloc(&d_in, in_bytes);
    cudaMalloc(&d_partial_max, partial_max_bytes);
    cudaMalloc(&d_full_max, full_max_bytes);
    cudaMalloc(&d_exp, exp_bytes);
    cudaMalloc(&d_partial_sum, partial_sum_bytes);
    cudaMalloc(&d_full_sum, full_sum_bytes);
    cudaMalloc(&d_out, out_bytes);

    // copy data to GPU
    cudaMemcpy(d_in, h_in, in_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_partial_max, h_partial_max, partial_max_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_full_max, h_full_max, full_max_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_exp, h_exp, exp_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_partial_sum, h_partial_sum, partial_sum_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_full_sum, h_full_sum, full_sum_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_out, h_out, out_bytes, cudaMemcpyHostToDevice);

    // grid and block dims
    dim3 partialGridDim(size / (BLOCKSIZE * 2), 1, 1);
    dim3 partialBlockDim(BLOCKSIZE);

    dim3 fullGridDim(1, 1, 1);
    dim3 fullBlockDim(BLOCKSIZE);

    dim3 normalizeGridDim(size / BLOCKSIZE, 1, 1);
    dim3 normalizeBlockDim(BLOCKSIZE);
    
    // launch partial max kernel and fetch results
    partial_max<<<partialGridDim, partialBlockDim>>>(d_in, d_partial_max);
    cudaDeviceSynchronize();

    // launch full max kernel
    full_max<<<fullGridDim, fullBlockDim>>>(d_partial_max, d_full_max);
    cudaDeviceSynchronize();

    // launch exp_sum kernel
    exp_sum<<<partialGridDim, partialBlockDim>>>(d_in, d_exp, d_partial_sum, d_full_max);
    cudaDeviceSynchronize();

    // launch full_sum kernel
    full_sum<<<fullGridDim, fullBlockDim>>>(d_partial_sum, d_full_sum);
    cudaDeviceSynchronize();

    // launch normalize kernel
    normalize<<<normalizeGridDim, normalizeBlockDim>>>(d_exp, d_out, d_full_sum);
    cudaDeviceSynchronize();

    cudaMemcpy(h_out, d_out, out_bytes, cudaMemcpyDeviceToHost);

    std::cout << "Idx 0 " << *h_out << std::endl;

    softmax_check(h_in, h_out);

    // free memory
    free(h_in);
    free(h_partial_max);
    free(h_full_max);
    free(h_exp);
    free(h_partial_sum);
    free(h_full_sum);
    free(h_out);
    cudaFree(d_in);
    cudaFree(d_partial_max);
    cudaFree(d_full_max);
    cudaFree(d_exp);
    cudaFree(d_partial_sum);
    cudaFree(d_full_sum);
    cudaFree(d_out);
    
    return 0;
}