#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>

__global__ void max_reduce(float *in, float *out) {
    __shared__ float sdata[128];

    // since half of the threads are idle, halve the number of blocks
    // two loads instead of one and perform first addition of the reduction
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x; // halve the amount of blocks
    sdata[tid] = max(in[idx], in[idx + blockDim.x]); // max of elements that are a block away from eachother
    
    __syncthreads();

    // reduction
    // reverse iteration: add elements that are blockDim.x / 2 away from each other, then halve the arr
    // offset is the distance between addends
    for (int offset = blockDim.x / 2; offset > 16; offset /= 2) {
        if(tid < offset) {
            sdata[tid] = max(sdata[tid], sdata[tid + offset]);
        }

        __syncthreads();
    }

    // implement shuffling - once it's down to 32 threads then everything is within the same warp
    // starts at 16 since each addend is half the distance of the amount of non-idle threads
    float reg = sdata[tid];
    int delta = 16;

    for (int i = 0; i < 5; i++) {
        reg = max(reg, __shfl_down_sync(0xffffffff, reg, delta, 32));
        delta /= 2;
    }

    // write
    if (tid == 0) {
        out[blockIdx.x] = reg; // write data into index that corresponds with blockIdx, 0 ends up holding our final sum
    }
}