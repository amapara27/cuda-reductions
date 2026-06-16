#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>

#define SIZE 32768
#define BLOCKSIZE 128

// one pass softmax kernel for 32 elements to start
__global__ void softmax(float *in, float *out_m, float *out_s) {
    // each thread grabs an element
    int tid = threadIdx.x;
    float element = in[tid];

    float m = element; // the max of one element is just the element
    float s = 1; // sum of one element is just e^(element - max) = 1

    // merge
    for (int delta = 16; delta > 0; delta /= 2) {
        // grab next pair
        float m_next = __shfl_down_sync(0xffffffff, m, delta, 32);
        float s_next = __shfl_down_sync(0xffffffff, s, delta, 32);

        float m_new = fmaxf(m, m_next); // find new max
        s = expf(m - m_new) * s + expf(m_next - m_new) * s_next; // calculate new exp_sum by scaling the current thread's and neighbors 

        // set max to the newest (largest) max
        m = m_new;
    }

    if (tid == 0) {
        *out_m = m;
        *out_s = s;
    }
}
