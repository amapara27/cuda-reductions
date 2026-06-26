#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <cmath>

#include "../../kernels/layer_norm.cuh"

#define SIZE 32768
#define BLOCKSIZE 1024
#define EPS 1e-5f

// array initialization
void arr_init(float *arr_in, float *arr_out, int size) {
    for (int i = 0; i < size; ++i) {
        arr_in[i] = (float)(rand() % 100);
        arr_out[i] = 0;
    }
}

// layernorm checker
void layernorm_check(float *cpu_in, float *gpu_out) {
    float mean = 0;
    for (int i = 0; i < SIZE; ++i) {
        mean += cpu_in[i];
    }
    mean /= SIZE;

    float var = 0;
    for (int i = 0; i < SIZE; ++i) {
        float d = cpu_in[i] - mean;
        var += d * d;
    }
    var /= SIZE;

    float inv_std = 1.0f / sqrtf(var + EPS);

    // compare elementwise
    bool correct = true;
    int idx = 0;
    for (int i = 0; i < SIZE; ++i) {
        float ref = (cpu_in[i] - mean) * inv_std;

        if (fabs(gpu_out[i] - ref) / (fabs(ref) + 1e-4f) > 1e-3f) {
            correct = false;
            idx = i;
            break;
        }
    }

    if (correct) {
        std::cout << "Correct! mean: " << mean << " var: " << var << std::endl;
    } else {
        float ref = (cpu_in[idx] - mean) * inv_std;
        std::cout << "Wrong at idx " << idx
                  << " GPU: " << gpu_out[idx]
                  << " CPU: " << ref
                  << " Diff: " << gpu_out[idx] - ref << std::endl;
    }
}

int main() {
    int size = SIZE;

    size_t bytes = size * sizeof(float);

    // host alloc
    float *h_in  = (float*)malloc(bytes);
    float *h_out = (float*)malloc(bytes);

    arr_init(h_in, h_out, size);

    // device alloc
    float *d_in, *d_out;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);

    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);

    // one block reduces the whole array
    dim3 gridDim(1, 1, 1);
    dim3 blockDim(BLOCKSIZE);

    // timer
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    layer_norm<<<gridDim, blockDim>>>(d_in, d_out);
    cudaEventRecord(stop);

    cudaDeviceSynchronize();

    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    std::cout << "Kernel time: " << ms << " ms" << std::endl;

    layernorm_check(h_in, h_out);

    // cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    free(h_in);
    free(h_out);
    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}
