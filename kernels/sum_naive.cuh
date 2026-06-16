#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <iostream>

__global__ void sum(float *in, float *out) {
    __shared__ float sdata[128];

    // since half of the threads are idle, halve the number of blocks
    // two loads instead of one and perform first addition of the reduction
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x; // halve the amount of blocks
    sdata[tid] = in[idx] + in[idx + blockDim.x]; // first addition - the sdata will have blockDim.x elements
    
    __syncthreads();

    // reduction
    // reverse iteration: add elements that are blockDim.x / 2 away from each other, then halve the arr
    for (int i = blockDim.x / 2; i > 0; i /= 2) {
        if(tid < i) {
            sdata[tid] += sdata[tid + i];
        }

        __syncthreads();
    }

    // write
    if (tid == 0) {
        out[blockIdx.x] = sdata[0]; // write data into index that corresponds with blockIdx, 0 ends up holding our final sum for each block
    }
}