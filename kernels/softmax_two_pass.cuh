#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>

#define SIZE 32768
#define BLOCKSIZE 128

// calculates partial max for each block
__global__ void partial_max(float *in, float *out) {
    // smem for the first round of maxes
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

    for (int i = 0; i < 5; ++i) {
        reg = max(reg, __shfl_down_sync(0xffffffff, reg, delta, 32));
        delta /= 2;
    }

    // write
    if (tid == 0) {
        out[blockIdx.x] = reg; // write data into index that corresponds with blockIdx, 0 ends up holding our final sum
    }
}

// calculates the overall max from the partial maxes
__global__ void full_max(float *in, float *out) {
    // allocate smem for the partial maxes
    __shared__ float sdata[128];

    unsigned int tid = threadIdx.x;
    sdata[tid] = in[tid];

    __syncthreads();

    // start offset at half the size of the partial max length
    for (int offset = SIZE / (blockDim.x * 2) / 2; offset > 16; offset /= 2) {
        if (tid < offset) {
            sdata[tid] = max(sdata[tid], sdata[tid + offset]);
        }
        __syncthreads();
    }

    // implement shuffling - once it's down to 32 threads then everything is within the same warp
    // starts at 16 since each addend is half the distance of the amount of non-idle threads
    float reg = sdata[tid];
    int delta = 16;

    for (int i = 0; i < 5; ++i) {
        reg = max(reg, __shfl_down_sync(0xffffffff, reg, delta, 32));
        delta /= 2;
    }

    // write
    if (tid == 0) {
        *out = reg; // write data into index that corresponds with blockIdx, 0 ends up holding our final sum
    }
}

__global__ void exp_sum(float *in, float *exp_out, float *partial_sum, float *max) {
    // allocate smem 
    __shared__ float sdata[BLOCKSIZE];

    // indexes
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x; // halve the amount of blocks to maximize thread usage

    // fill exponential array
    float exp1 = expf(in[idx] - *max);
    float exp2 = expf(in[idx + blockDim.x] - *max);

    exp_out[idx] = exp1;
    exp_out[idx + blockDim.x] = exp2;

    // write sum of exponentials - first load
    sdata[tid] = exp1 + exp2;

    __syncthreads();

    // reduction - sum the exponentials
    for (int offset = blockDim.x / 2; offset > 16; offset /= 2) {
        if (tid < offset) {
            sdata[tid] = sdata[tid] + sdata[tid + offset];
        }
        __syncthreads();
    }

    // implement shuffling - once it's down to 32 threads then everything is within the same warp
    // starts at 16 since each addend is half the distance of the amount of non-idle threads
    float reg = sdata[tid];
    int delta = 16;

    for (int i = 0; i < 5; ++i) {
        reg += __shfl_down_sync(0xffffffff, reg, delta, 32);
        delta /= 2;
    }

    if (tid == 0) {
        partial_sum[blockIdx.x] = reg;
    }
}

// calculates the overall max from the partial sums
__global__ void full_sum(float *in, float *out) {
    // allocate smem for the partial sums
    __shared__ float sdata[128];

    unsigned int tid = threadIdx.x;
    sdata[tid] = in[tid];

    __syncthreads();

    // start offset at half the size of the partial max length
    for (int offset = SIZE / (blockDim.x * 2) / 2; offset > 16; offset /= 2) {
        if (tid < offset) {
            sdata[tid] = sdata[tid] + sdata[tid + offset];
        }
        __syncthreads();
    }

    // implement shuffling - once it's down to 32 threads then everything is within the same warp
    // starts at 16 since each addend is half the distance of the amount of non-idle threads
    float reg = sdata[tid];
    int delta = 16;

    for (int i = 0; i < 5; ++i) {
        reg += __shfl_down_sync(0xffffffff, reg, delta, 32);
        delta /= 2;
    }

    // write
    if (tid == 0) {
        *out = reg; // write data into index that corresponds with blockIdx, 0 ends up holding our final sum
    }
}


// normalizes our array
__global__ void normalize(float *in, float *out, float *sum) {
    int idx = blockIdx.x *  blockDim.x + threadIdx.x;
    out[idx] = in[idx] / *sum;
}