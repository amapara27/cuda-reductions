#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>

// one pass softmax kernel for 32 elements to start
__global__ void softmax(float *in, float *out) {
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

    // only thread 0 has the correct max and sum, must broadcast them to all within warp
    m = __shfl_sync(0xffffffff, m, 0, 32);
    s = __shfl_sync(0xffffffff, s, 0, 32);

    out[tid] = expf(element - m)  / s;
}
