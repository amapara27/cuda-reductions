#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <algorithm>

#include "../../kernels/sum_naive.cuh" 
#include "../../kernels/sum_shuffle.cuh" 
#include "../../kernels/max_reduce.cuh" 
#include "../../kernels/softmax_two_pass.cuh" 

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
    return 0;
}