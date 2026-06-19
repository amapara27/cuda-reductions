#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <cfloat>

#define SIZE 32768
#define BLOCKSIZE 128

// one pass softmax kernel
__global__ void softmax(float *in, float *out) {
    // allocate smem: size / warp size
    __shared__ float m_arr[BLOCKSIZE / 32];
    __shared__ float s_arr[BLOCKSIZE / 32];

    // each thread grabs an element
    int tid = threadIdx.x;

    // set max and sum values to minimums before iteration
    float m = -FLT_MAX;
    float s = 0;

    // each thread grabs one element in a block than moves onto next block
    for (int i = tid; i < SIZE ; i += blockDim.x) {
        float x = in[i];
        float m_new = fmaxf(m, x);
        s = expf(m - m_new) * s + expf(x - m_new); // correct prev sum and add current exponential (s_next is 1)
        m = m_new; // set new max
    }

    // merge the sum and max vals for each warp into lane 0
    for (int delta = 16; delta > 0; delta /= 2) {
        // grab next pair
        float m_next = __shfl_down_sync(0xffffffff, m, delta, 32);
        float s_next = __shfl_down_sync(0xffffffff, s, delta, 32);

        float m_new = fmaxf(m, m_next); // find new max
        s = expf(m - m_new) * s + expf(m_next - m_new) * s_next; // calculate new exp_sum by scaling the current thread's and neighbors 

        // set max to the newest (largest) max
        m = m_new;
    }

    int warp_id = tid / 32;
    int lane_id = tid % 32;

    // lane 0 within each warp holds the full sum and max values of that warp
    if (lane_id == 0) {
        m_arr[warp_id] = m;
        s_arr[warp_id] = s;
    }

    __syncthreads();

    int num_warps = blockDim.x / 32;

    // one warp merges all the results
    if (warp_id == 0) {
        // initialize m and s vals - only using num_warps lanes
        m = (lane_id < num_warps) ? m_arr[lane_id] : -FLT_MAX;
        s = (lane_id < num_warps) ? s_arr[lane_id] : 0;

        for (int offset = 16; offset > 0; offset /= 2) {
            // grab next vals
            float m_next = __shfl_down_sync(0xffffffff, m, offset, 32);
            float s_next = __shfl_down_sync(0xffffffff, s, offset, 32);

            // new max
            float m_new = fmaxf(m, m_next);

            // scale this lane's, next one's sum, and add them
            s = expf(m - m_new) * s + expf(m_next - m_new) * s_next;

            m = m_new;
        }
    }

    // write final max and sum vals to idx 0 of arrs (lane 0 holds)
    if (tid == 0) {
        m_arr[0] = m;
        s_arr[0] = s;
    }

    __syncthreads();

    float m_final = m_arr[0];
    float s_final = s_arr[0];

    // normalization for multi-element threads
    for (int i = tid; i < SIZE; i += blockDim.x) {
        out[i] = expf(in[i] - m_final) / s_final;
    }
}
