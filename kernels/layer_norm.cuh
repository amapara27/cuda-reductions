#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>

#define SIZE 32768
#define BLOCKSIZE 1024

__global__ void layer_norm(float *in, float *out) {
    // set thread id
    int tid = threadIdx.x;

    // values
    int count = 0;
    float mean = 0;
    float M2 = 0;

    // per-thread iteration
    for (int i = tid; i < SIZE; i += blockDim.x) {
        float x = in[i];

        // welford's 
        count += 1;
        float delta = x - mean; 
        mean += delta / count; // updates mean incrementally
        float delta2 = x - mean; // updated delta
        M2 += delta * delta2; // squared deviations, update previous
    }

    // merge
    for (int offset = 16; offset > 0; offset /= 2) {
        // grab next vals
        int count_next = __shfl_down_sync(0xffffffff, count, offset, 32);
        float mean_next = __shfl_down_sync(0xffffffff, mean, offset, 32);
        float M2_next = __shfl_down_sync(0xffffffff, M2, offset, 32);

        int count_curr = count;
        float mean_curr = mean;
        float M2_curr = M2;

        count = count_curr + count_next; // total count
        float delta = mean_next - mean_curr; // delta is diff between means
        mean = mean_curr + delta * ((float)count_next / count); // update mean - nudges towards B
        M2 = (M2_curr + M2_next) + delta * delta * ((float)count_curr * count_next / count); // squared deviations: spread within each group + spread between grups
    }
}